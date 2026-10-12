// hyprliquid — a window you drag becomes a drop of liquid glass.
//
// Picking a window up (SUPER + drag) pours it into a liquid body of a fixed
// size that hangs from the cursor:
//   * the body is a soft-body mesh (shape matching, Müller et al. 2005) held
//     at the spot you grabbed: it lags, stretches, swings and wobbles;
//   * its surface is a small wave simulation: moving stirs it, the liquid
//     sloshes back and forth when you start, stop or turn;
//   * it is rendered as liquid glass: the window's content seen through the
//     surface (refracted, with dispersion), a thick clear rim that bends what
//     lies behind it (red and blue bend differently), a meniscus that pulls
//     the background in around it, specular light and a thin-film shimmer;
//   * fast moves shed droplets that trail behind and evaporate.
// Dropping it pours the liquid into the window's new place and the real
// window takes over. Workspace switches and window animations are left to
// Hyprland.

#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/desktop/Workspace.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>
#include <hyprland/src/helpers/math/Math.hpp>
#include <hyprland/src/layout/LayoutManager.hpp>
#include <hyprland/src/pointer/PointerManager.hpp>
#include <hyprland/src/managers/input/InputManager.hpp>

#include <GLES3/gl32.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

static HANDLE PHANDLE = nullptr; // not inline: a GNU-unique symbol would pin the .so on unload

namespace {
    // body
    constexpr double BODY_HALF   = 160.0; // half the longer side of the liquid body, logical px
    constexpr double HALO        = 30.0;  // meniscus around it, logical px
    constexpr double RIM         = 22.0;  // thick clear glass rim, logical px
    constexpr int    NX          = 34;    // mesh vertices across (incl. the halo ring)
    constexpr int    NY          = 26;
    constexpr float  POUR_IN     = 0.34f; // s: window -> drop
    constexpr float  POUR_OUT    = 0.36f; // s: drop -> window
    constexpr float  RELEASE_END = 0.55f; // s: real window back
    // soft body
    constexpr float K_SHAPE = 520.f;  // pull toward the matched rest shape (1/s^2)
    constexpr float K_GRAB  = 2600.f; // the grabbed spot follows the cursor
    constexpr float K_HOME  = 900.f;  // on release: pull into the window's place
    constexpr float DAMP    = 7.5f;   // 1/s
    constexpr float UPRIGHT = 0.90f;  // how much of its tilt the body keeps per match (it rights itself)
    constexpr float SUBSTEP = 1.f / 480.f;
    // surface waves
    constexpr int   WX = 96, WY = 64;
    constexpr float WAVE_STEP = 1.f / 120.f;
    // droplets
    constexpr size_t DROPS_MAX = 160;

    using Clock = std::chrono::steady_clock;

    struct V2 {
        float x = 0, y = 0;
    };
    inline V2 operator+(V2 a, V2 b) {
        return {a.x + b.x, a.y + b.y};
    }
    inline V2 operator-(V2 a, V2 b) {
        return {a.x - b.x, a.y - b.y};
    }
    inline V2 operator*(V2 a, float s) {
        return {a.x * s, a.y * s};
    }
    inline float len(V2 a) {
        return std::sqrt(a.x * a.x + a.y * a.y);
    }

    struct SDrop {
        V2    pos, vel;
        float r, life, maxLife;
        V2    uv; // where on the window it came from (its colour)
    };

    enum ePhase : uint8_t {
        IDLE,
        HOLD,
        RELEASE
    };

    struct SMonitorState {
        ePhase                   phase = IDLE;
        PHLWINDOWREF             dragWin, hiddenWin;
        SP<Render::IFramebuffer> snap, scene;
        Clock::time_point        phaseStart, lastStep, lastWave;
        // soft body (GL px, rows top-down)
        std::vector<V2> x, v;
        std::vector<V2> grid;            // rest position of each vertex in [-1,1] of the extended box
        V2              winHalf, bodyHalf, startHalf, targetHalf; // body half size: now / start / target
        V2              grabN;           // grabbed spot, [-1,1] of the body
        V2              homeC;           // release: centre of the window's new place
        float           corner = 0.f, glass = 0.f, scale = 1.f;
        V2              prevC, prevVel, vel;
        // sloshing: a damped 2D oscillator driven by the body's acceleration
        V2 slosh, sloshV;
        // waves
        std::vector<float>   h, hv;
        std::vector<uint8_t> waveTex;
        float                nextRain = 0.f;
        // droplets
        std::vector<SDrop> drops;
        std::mt19937       rng{0x11};
        // GL objects
        GLuint vao = 0, posVbo = 0, uvVbo = 0, ebo = 0, waveId = 0, dropVao = 0, dropVbo = 0;
        GLsizei indexCount = 0;
    };

    std::unordered_map<MONITORID, SMonitorState> g_mon;
    std::vector<CHyprSignalListener>            g_listeners;

    // ---------------------------------------------------------------- GL ---

    const char* LIQUID_VERT = R"(#version 300 es
precision highp float;
layout(location = 0) in vec2 pos;   // GL px
layout(location = 1) in vec2 grid;  // rest position in [-1,1] of the extended box
uniform vec2 fbSize;
out vec2 nUv;
void main() {
    nUv         = grid;
    gl_Position = vec4(pos / fbSize * 2.0 - 1.0, 0.0, 1.0);
}
)";

    const char* LIQUID_FRAG = R"(#version 300 es
precision highp float;
in vec2 nUv;
uniform sampler2D win;      // the window when it was picked up
uniform sampler2D scene;    // the screen behind the liquid
uniform sampler2D waves;    // surface height, 0.5 = rest
uniform vec2  fbSize;
uniform vec2  extHalf;      // rest half size incl. the halo
uniform vec2  bodyHalf;
uniform float corner;
uniform float rim;          // px
uniform float halo;         // px
uniform float px;           // monitor scale (1 logical px)
uniform vec2  slosh;        // surface tilt
uniform float time;
uniform float glass;        // 0 = plain window .. 1 = full liquid glass
out vec4 frag;

float sdRound(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

vec3 sceneRGB(vec2 at, vec2 shift, float spread) {
    // one bend for green, a bit more for red, a bit less for blue: dispersion
    return vec3(texture(scene, (at + shift * (1.0 + spread)) / fbSize).r,
                texture(scene, (at + shift) / fbSize).g,
                texture(scene, (at + shift * (1.0 - spread)) / fbSize).b);
}

void main() {
    vec2  p   = nUv * extHalf;
    float d   = -sdRound(p, bodyHalf, corner);   // > 0 inside the body (rest px)
    vec2  spx = gl_FragCoord.xy;
    vec2  gd  = vec2(dFdx(d), dFdy(d));
    vec2  n   = length(gd) > 1e-6 ? normalize(gd) : vec2(0.0, 1.0); // screen: pointing into the body
    float ppx = max(length(gd), 1e-3);            // rest px per screen px (the mesh stretches)
    vec3  L   = normalize(vec3(-0.45, -0.70, 0.55)); // light from the upper left (rows run top-down)

    if (d < 0.0) {
        // the meniscus: the background right next to the liquid is pulled in
        // toward it and darkens a little where the drop touches the glass
        float o = -d / (halo * ppx);
        if (o >= 1.0)
            discard;
        float prof = (1.0 - o) * (1.0 - o) * (1.0 - o);
        vec3  col  = sceneRGB(spx, n * 16.0 * px * prof * glass, 0.35);
        col       *= 1.0 - 0.22 * prof * glass;
        // a faint caustic where light focused through the drop lands
        col += vec3(0.10) * glass * exp(-pow((o - 0.35) / 0.18, 2.0)) * max(dot(n, normalize(L.xy)), 0.0);
        frag = vec4(col, 1.0);
        return;
    }

    // surface: waves + slosh tilt
    vec2  wuv = p / (2.0 * bodyHalf) + 0.5;
    vec2  tx  = 1.0 / vec2(textureSize(waves, 0));
    float h   = texture(waves, wuv).r * 2.0 - 1.0;
    vec2  g   = vec2(texture(waves, wuv + vec2(tx.x, 0.0)).r - texture(waves, wuv - vec2(tx.x, 0.0)).r,
                     texture(waves, wuv + vec2(0.0, tx.y)).r - texture(waves, wuv - vec2(0.0, tx.y)).r) * 2.0;
    g += slosh;
    // the rim domes down to the edge
    float e  = 1.0 - clamp(d / (rim * ppx), 0.0, 1.0);
    float e2 = e * e;
    vec3  N  = normalize(vec3((-n * e2 * e * 2.4 - g * 2.2) * glass, 1.0));

    // the window seen through the liquid: refracted by the surface, magnified
    // toward the rim like through a lens, with dispersion
    vec2 cuv  = wuv;
    cuv       = 0.5 + (cuv - 0.5) * (1.0 - 0.10 * e2 * glass);
    vec2 refr = -N.xy * 0.045 * glass;
    vec3 content = vec3(texture(win, cuv + refr * 1.35).r, texture(win, cuv + refr).g, texture(win, cuv + refr * 0.65).b);

    // the clear rim: what lies behind, bent hard toward the inside
    vec3  behind = sceneRGB(spx, -n * (46.0 * px) * e2 - N.xy * 20.0 * px, 0.30);
    float clear  = smoothstep(0.30, 0.95, e) * 0.92 * glass;
    vec3  col    = mix(content, behind, clear);
    // depth: the liquid gets a touch darker deeper in the rim's shadow
    col *= 1.0 - 0.10 * glass * smoothstep(0.0, 0.35, e) * (1.0 - smoothstep(0.35, 0.9, e));

    // light
    vec3  H    = normalize(L + vec3(0.0, 0.0, 1.0));
    float nh   = max(dot(N, H), 0.0);
    float spec = pow(nh, 90.0) * 1.2 + pow(nh, 14.0) * 0.10;
    float fres = pow(1.0 - N.z, 1.6);
    col += vec3(spec + fres * 0.55) * glass;
    // a thin bright line just inside the edge, strongest facing the light
    float line = exp(-pow((d - 1.6 * px * ppx) / (1.1 * px * ppx), 2.0));
    col += vec3(0.55) * line * (0.25 + 0.75 * max(dot(-n, normalize(L.xy)), 0.0)) * glass;
    // thin-film shimmer on the dome
    vec3 film = 0.5 + 0.5 * cos(6.2831853 * (e * 1.4 + h * 0.6 + time * 0.12 + vec3(0.0, 0.33, 0.67)));
    col += film * e2 * e2 * 0.20 * glass;
    frag = vec4(col, 1.0);
}
)";

    const char* DROP_VERT = R"(#version 300 es
precision highp float;
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 drop;   // centre (GL px), radius, life left 0..1
layout(location = 2) in vec2 tint;   // window uv it came from
uniform vec2 fbSize;
out vec2  q;
out float r;
out float life;
flat out vec2 tintUv;
void main() {
    q      = corner * 2.0 - 1.0;
    r      = drop.z;
    life   = drop.w;
    tintUv = tint;
    vec2 p = drop.xy + q * (drop.z + 2.0);
    gl_Position = vec4(p / fbSize * 2.0 - 1.0, 0.0, 1.0);
}
)";

    const char* DROP_FRAG = R"(#version 300 es
precision highp float;
in vec2  q;
in float r;
in float life;
flat in vec2 tintUv;
uniform sampler2D scene;
uniform sampler2D win;
uniform vec2 fbSize;
out vec4 frag;
void main() {
    float rho = length(q) * (r + 2.0) / r;
    if (rho > 1.0)
        discard;
    vec3 N = vec3(q * (r + 2.0) / r, sqrt(max(1.0 - rho * rho, 0.0)));
    // a bead of glass: the world behind it upside down and squeezed, with dispersion
    vec2 at = gl_FragCoord.xy;
    vec2 s  = -N.xy * r * 1.6;
    vec3 col = vec3(texture(scene, (at + s * 1.25) / fbSize).r, texture(scene, (at + s) / fbSize).g, texture(scene, (at + s * 0.75) / fbSize).b);
    col = mix(col, texture(win, tintUv).rgb, 0.35);
    col *= 0.70 + 0.30 * N.z;
    vec3  L    = normalize(vec3(-0.45, -0.70, 0.55));
    float spec = pow(max(dot(N, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 60.0);
    col += vec3(spec * 1.1);
    float a = smoothstep(1.0, 1.0 - 1.5 / r, rho) * smoothstep(0.0, 0.25, life);
    frag = vec4(col * a, a);
}
)";

    struct SGL {
        GLuint liquid = 0, drop = 0, quad = 0;
        GLint  lFb = -1, lExt = -1, lBody = -1, lCorner = -1, lRim = -1, lHalo = -1, lPx = -1, lSlosh = -1, lTime = -1, lGlass = -1, lWin = -1, lScene = -1,
              lWaves = -1, lVFb = -1;
        GLint dFb = -1, dVFb = -1, dScene = -1, dWin = -1;
        bool  failed = false;
    } g_gl;

    GLuint compile(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024] = {};
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            HyprlandAPI::addNotification(PHANDLE, std::string("[hyprliquid] shader: ") + log, CHyprColor{1, 0.2, 0.2, 1}, 8000);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    GLuint link(const char* vsrc, const char* fsrc) {
        GLuint vs = compile(GL_VERTEX_SHADER, vsrc), fs = compile(GL_FRAGMENT_SHADER, fsrc);
        if (!vs || !fs) {
            if (vs)
                glDeleteShader(vs);
            if (fs)
                glDeleteShader(fs);
            return 0;
        }
        GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    bool ensureGL() {
        if (g_gl.liquid)
            return true;
        if (g_gl.failed)
            return false;
        g_gl.liquid = link(LIQUID_VERT, LIQUID_FRAG);
        g_gl.drop   = link(DROP_VERT, DROP_FRAG);
        if (!g_gl.liquid || !g_gl.drop) {
            g_gl.failed = true;
            return false;
        }
        const auto U = [](GLuint p, const char* n) { return glGetUniformLocation(p, n); };
        g_gl.lVFb    = U(g_gl.liquid, "fbSize");
        g_gl.lExt    = U(g_gl.liquid, "extHalf");
        g_gl.lBody   = U(g_gl.liquid, "bodyHalf");
        g_gl.lCorner = U(g_gl.liquid, "corner");
        g_gl.lRim    = U(g_gl.liquid, "rim");
        g_gl.lHalo   = U(g_gl.liquid, "halo");
        g_gl.lPx     = U(g_gl.liquid, "px");
        g_gl.lSlosh  = U(g_gl.liquid, "slosh");
        g_gl.lTime   = U(g_gl.liquid, "time");
        g_gl.lGlass  = U(g_gl.liquid, "glass");
        g_gl.lWin    = U(g_gl.liquid, "win");
        g_gl.lScene  = U(g_gl.liquid, "scene");
        g_gl.lWaves  = U(g_gl.liquid, "waves");
        g_gl.dVFb    = U(g_gl.drop, "fbSize");
        g_gl.dScene  = U(g_gl.drop, "scene");
        g_gl.dWin    = U(g_gl.drop, "win");

        GLint prevBuf = 0;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);
        glGenBuffers(1, &g_gl.quad);
        const float corners[] = {0, 0, 1, 0, 0, 1, 1, 1};
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.quad);
        glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, prevBuf);
        return true;
    }

    std::optional<GLuint> fbID(const SP<Render::IFramebuffer>& fb) {
        auto* gl = dynamic_cast<Render::GL::CGLFramebuffer*>(fb.get());
        if (!gl)
            return std::nullopt;
        return gl->getFBID();
    }

    bool ensureFB(SP<Render::IFramebuffer>& fb, int w, int h, DRMFormat fmt, const char* name) {
        if (fb && fb->m_size == Vector2D{(double)w, (double)h} && fb->m_drmFormat == fmt)
            return true;
        if (!fb)
            fb = g_pHyprRenderer->createFB(name);
        return fb->alloc(w, h, fmt);
    }

    // Copy a region of `from` to `to` at (0,0) (or all of it). Leaves bindings and scissor as they were.
    void blitRect(const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to, int x0, int y0, int x1, int y1) {
        const auto A = fbID(from), B = fbID(to);
        if (!A || !B)
            return;
        GLint prevRead = 0, prevDraw = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
        const GLboolean SCISSOR = glIsEnabled(GL_SCISSOR_TEST);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, *A);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, *B);
        glBlitFramebuffer(x0, y0, x1, y1, 0, 0, x1 - x0, y1 - y0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);
        if (SCISSOR)
            glEnable(GL_SCISSOR_TEST);
    }

    CBox glBox(const CBox& b, const PHLMONITOR& mon) {
        const double S = mon->m_scale;
        return CBox{(b.x - mon->m_position.x) * S, (b.y - mon->m_position.y) * S, b.w * S, b.h * S}; // rows run top-down
    }

    V2 cursorGL(const PHLMONITOR& mon) {
        const Vector2D C = Pointer::mgr()->position();
        return {(float)((C.x - mon->m_position.x) * mon->m_scale), (float)((C.y - mon->m_position.y) * mon->m_scale)};
    }

    float smooth01(float t) {
        t = std::clamp(t, 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    float secsSince(Clock::time_point t) {
        return std::chrono::duration<float>(Clock::now() - t).count();
    }

    // ---------------------------------------------------------- the body ---

    V2 haloPx(const SMonitorState& st) {
        return {(float)(HALO * st.scale), (float)(HALO * st.scale)};
    }

    // rest position of vertex i around the origin, for the current body size
    V2 restOf(const SMonitorState& st, size_t i) {
        const V2 E = st.bodyHalf + haloPx(st);
        return {st.grid[i].x * E.x, st.grid[i].y * E.y};
    }

    void stirWaves(SMonitorState& st, V2 atN, float amount, float radius) {
        // atN: [-1,1] of the body; a soft dent in the surface
        const float CX = (atN.x * 0.5f + 0.5f) * (WX - 1), CY = (atN.y * 0.5f + 0.5f) * (WY - 1);
        const int   R  = (int)std::ceil(radius * 2.f);
        for (int y = std::max(1, (int)CY - R); y <= std::min(WY - 2, (int)CY + R); ++y)
            for (int x = std::max(1, (int)CX - R); x <= std::min(WX - 2, (int)CX + R); ++x) {
                const float D2 = ((x - CX) * (x - CX) + (y - CY) * (y - CY)) / (radius * radius);
                st.hv[(size_t)y * WX + x] -= amount * std::exp(-D2);
            }
    }

    void stepWaves(SMonitorState& st) {
        std::vector<float>& h = st.h;
        std::vector<float>& v = st.hv;
        for (int y = 1; y < WY - 1; ++y)
            for (int x = 1; x < WX - 1; ++x) {
                const size_t I = (size_t)y * WX + x;
                v[I] += (h[I - 1] + h[I + 1] + h[I - WX] + h[I + WX] - 4.f * h[I]) * 0.45f;
                v[I] *= 0.986f;
            }
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = std::clamp(h[i] + v[i], -1.f, 1.f);
        // reflecting edges
        for (int x = 0; x < WX; ++x) {
            h[x]                         = h[x + WX];
            h[(size_t)(WY - 1) * WX + x] = h[(size_t)(WY - 2) * WX + x];
        }
        for (int y = 0; y < WY; ++y) {
            h[(size_t)y * WX]          = h[(size_t)y * WX + 1];
            h[(size_t)y * WX + WX - 1] = h[(size_t)y * WX + WX - 2];
        }
    }

    // Pick the window up: snapshot, hide it, build the mesh on the window.
    bool begin(SMonitorState& st, const PHLMONITOR& mon, const PHLWINDOW& w, const SP<Render::IFramebuffer>& cur) {
        CBox       B  = glBox(CBox{w->positionAnimation()->value(), w->sizeAnimation()->value()}, mon);
        const int  X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
        const int  X1 = std::min((int)cur->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)cur->m_size.y, (int)(B.y + B.h));
        if (X1 - X0 < 8 || Y1 - Y0 < 8 || !ensureFB(st.snap, X1 - X0, Y1 - Y0, cur->m_drmFormat, "hyprliquid-snap"))
            return false;
        blitRect(cur, st.snap, X0, Y0, X1, Y1);
        B = CBox{(double)X0, (double)Y0, (double)(X1 - X0), (double)(Y1 - Y0)};

        st.scale    = (float)mon->m_scale;
        st.winHalf  = {(float)B.w * 0.5f, (float)B.h * 0.5f};
        st.bodyHalf = st.startHalf = st.winHalf;
        const float R  = (float)(BODY_HALF * st.scale);
        const float MX = std::max(st.winHalf.x, st.winHalf.y);
        st.targetHalf  = {std::max(R * st.winHalf.x / MX, 0.55f * R), std::max(R * st.winHalf.y / MX, 0.55f * R)};

        const V2 C   = {(float)(B.x + B.w * 0.5), (float)(B.y + B.h * 0.5)};
        const V2 CUR = cursorGL(mon);
        st.grabN     = {std::clamp((CUR.x - C.x) / st.winHalf.x, -0.8f, 0.8f), std::clamp((CUR.y - C.y) / st.winHalf.y, -0.8f, 0.8f)};

        st.grid.clear();
        st.x.clear();
        st.v.clear();
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                st.grid.push_back({-1.f + 2.f * i / (NX - 1), -1.f + 2.f * j / (NY - 1)});
        for (size_t i = 0; i < st.grid.size(); ++i) {
            st.x.push_back(C + restOf(st, i));
            st.v.push_back({});
        }
        st.prevC = C;
        st.vel = st.prevVel = {};
        st.slosh = st.sloshV = {};
        st.h.assign((size_t)WX * WY, 0.f);
        st.hv.assign((size_t)WX * WY, 0.f);
        st.waveTex.assign((size_t)WX * WY, 128);
        stirWaves(st, st.grabN, 0.9f, 5.f); // the grab sends a ring through it
        st.drops.clear();
        st.glass    = 0.f;
        st.corner   = 10.f * st.scale;
        st.phase    = HOLD;
        st.phaseStart = st.lastStep = st.lastWave = Clock::now();
        st.nextRain = 0.3f;

        w->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(0.f); // the liquid stands in for it
        st.hiddenWin = w;
        return true;
    }

    void showWindow(SMonitorState& st) {
        if (const auto W = st.hiddenWin.lock())
            W->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(1.f);
        st.hiddenWin.reset();
    }

    void stepBody(SMonitorState& st, const V2 cursor, float dt) {
        const size_t N = st.x.size();
        // match the rest shape to where the vertices are (translation + rotation)
        V2 c{};
        for (const auto& p : st.x)
            c = c + p;
        c = c * (1.f / N);
        float a00 = 0, a01 = 0, a10 = 0, a11 = 0;
        for (size_t i = 0; i < N; ++i) {
            const V2 D = st.x[i] - c, Q = restOf(st, i);
            a00 += D.x * Q.x;
            a01 += D.x * Q.y;
            a10 += D.y * Q.x;
            a11 += D.y * Q.y;
        }
        const float TH = std::atan2(a10 - a01, a00 + a11) * UPRIGHT;
        const float CS = std::cos(TH), SN = std::sin(TH);

        const bool  HOLDING = st.phase == HOLD;
        const float HOMEK   = HOLDING ? 0.f : K_HOME * smooth01(secsSince(st.phaseStart) / (POUR_OUT * 0.6f));
        const V2    GQ      = {st.grabN.x * st.bodyHalf.x, st.grabN.y * st.bodyHalf.y};
        const float SIG2    = std::pow(0.45f * std::min(st.bodyHalf.x, st.bodyHalf.y), 2.f) * 2.f;
        for (size_t i = 0; i < N; ++i) {
            const V2 Q    = restOf(st, i);
            const V2 GOAL = c + V2{CS * Q.x - SN * Q.y, SN * Q.x + CS * Q.y};
            V2       acc  = (GOAL - st.x[i]) * K_SHAPE - st.v[i] * DAMP;
            if (HOLDING) {
                const V2    DQ = Q - GQ;
                const float W  = std::exp(-(DQ.x * DQ.x + DQ.y * DQ.y) / SIG2);
                acc            = acc + ((cursor + DQ) - st.x[i]) * (K_GRAB * W);
            } else
                acc = acc + ((st.homeC + Q) - st.x[i]) * HOMEK;
            st.v[i] = st.v[i] + acc * dt;
        }
        for (size_t i = 0; i < N; ++i)
            st.x[i] = st.x[i] + st.v[i] * dt;
    }

    void simulate(SMonitorState& st, const PHLMONITOR& mon) {
        const float T  = secsSince(st.phaseStart);
        const float FRAME = std::min(secsSince(st.lastStep), 1.f / 30.f);
        float       DT    = FRAME;
        st.lastStep       = Clock::now();
        const V2 CUR   = cursorGL(mon);

        // size, corners and glass follow the phase
        if (st.phase == HOLD) {
            const float K = smooth01(T / POUR_IN);
            st.bodyHalf   = st.startHalf + (st.targetHalf - st.startHalf) * K;
            st.corner     = 10.f * st.scale + (0.42f * std::min(st.targetHalf.x, st.targetHalf.y) - 10.f * st.scale) * K;
            st.glass      = smooth01(T / (POUR_IN * 0.7f));
        } else {
            const float K = smooth01(T / POUR_OUT);
            st.bodyHalf   = st.startHalf + (st.targetHalf - st.startHalf) * K;
            st.corner     = 0.42f * std::min(st.startHalf.x, st.startHalf.y) + (10.f * st.scale - 0.42f * std::min(st.startHalf.x, st.startHalf.y)) * K;
            st.glass      = 1.f - smooth01((T - POUR_OUT * 0.35f) / (RELEASE_END - POUR_OUT * 0.35f));
        }

        while (DT > 0.f) {
            const float H = std::min(DT, SUBSTEP);
            stepBody(st, CUR, H);
            DT -= H;
        }

        // body motion -> slosh (a damped oscillator pushed by the acceleration)
        V2 c{};
        for (const auto& p : st.x)
            c = c + p;
        c                = c * (1.f / st.x.size());
        const float STEP = std::max(FRAME, 1e-3f);
        const V2    VEL  = (c - st.prevC) * (1.f / std::max(STEP, 1e-3f));
        st.prevC         = c;
        const V2 ACC     = (VEL - st.prevVel) * (1.f / std::max(STEP, 1e-3f));
        st.prevVel       = VEL;
        st.vel           = VEL;
        const float R    = std::max(st.bodyHalf.x, st.bodyHalf.y);
        const V2    PUSH = ACC * (-0.00018f / std::max(R / 240.f, 0.3f));
        st.sloshV        = st.sloshV + (PUSH - st.slosh * 140.f - st.sloshV * 5.f) * STEP;
        st.slosh         = st.slosh + st.sloshV * STEP;
        st.slosh         = {std::clamp(st.slosh.x, -0.5f, 0.5f), std::clamp(st.slosh.y, -0.5f, 0.5f)};

        // waves: stirred where you hold it (by speed), the odd drop of "rain"
        float WT = secsSince(st.lastWave);
        if (WT >= WAVE_STEP) {
            st.lastWave = Clock::now();
            std::uniform_real_distribution<float> uni(0.f, 1.f);
            const float SPEED = len(VEL) / (st.scale * 1000.f); // ~1 at 1000 logical px/s
            if (st.phase == HOLD)
                stirWaves(st, st.grabN, std::min(SPEED, 2.f) * 0.08f, 3.5f);
            st.nextRain -= WT;
            if (st.nextRain <= 0.f) {
                stirWaves(st, {uni(st.rng) * 1.6f - 0.8f, uni(st.rng) * 1.6f - 0.8f}, 0.25f, 2.5f);
                st.nextRain = 0.25f + 0.5f * uni(st.rng);
            }
            for (int k = 0; k < std::min(4, (int)(WT / WAVE_STEP)); ++k)
                stepWaves(st);
            for (size_t i = 0; i < st.h.size(); ++i)
                st.waveTex[i] = (uint8_t)std::clamp((int)std::lround((st.h[i] * 0.5f + 0.5f) * 255.f), 0, 255);

            // droplets: shed from the trailing edge on fast moves
            if (st.phase == HOLD && T > POUR_IN && SPEED > 0.6f && st.drops.size() < DROPS_MAX) {
                const int COUNT = (int)std::min(3.f, SPEED * 1.2f * uni(st.rng) + (uni(st.rng) < SPEED * 0.5f ? 1.f : 0.f));
                const V2  DIR   = VEL * (1.f / std::max(len(VEL), 1e-3f));
                for (int k = 0; k < COUNT; ++k) {
                    // the body vertex on the edge farthest back
                    size_t best = 0;
                    float  bestDot = 1e9f;
                    for (size_t i = 0; i < st.x.size(); ++i) {
                        const V2 Q = restOf(st, i);
                        if (std::abs(std::abs(Q.x) - st.bodyHalf.x) > 0.12f * st.bodyHalf.x && std::abs(std::abs(Q.y) - st.bodyHalf.y) > 0.12f * st.bodyHalf.y)
                            continue;
                        if (std::abs(Q.x) > st.bodyHalf.x * 1.02f || std::abs(Q.y) > st.bodyHalf.y * 1.02f)
                            continue;
                        const V2    D   = st.x[i] - c;
                        const float DOT = D.x * DIR.x + D.y * DIR.y + (uni(st.rng) - 0.5f) * R * 0.9f;
                        if (DOT < bestDot) {
                            bestDot = DOT;
                            best    = i;
                        }
                    }
                    const V2    Q    = restOf(st, best);
                    const V2    PERP = {-DIR.y, DIR.x};
                    const float LIFE = 0.35f + 0.45f * uni(st.rng);
                    st.drops.push_back(SDrop{st.x[best], VEL * (0.25f + 0.3f * uni(st.rng)) + PERP * ((uni(st.rng) - 0.5f) * 260.f * st.scale),
                                             (2.5f + 6.f * uni(st.rng) * uni(st.rng)) * st.scale, LIFE, LIFE,
                                             V2{Q.x / (2.f * st.bodyHalf.x) + 0.5f, Q.y / (2.f * st.bodyHalf.y) + 0.5f}});
                }
            }
        }
        // droplets fly on, slow down and evaporate
        const float FT = FRAME;
        for (auto& d : st.drops) {
            d.pos  = d.pos + d.vel * FT;
            d.vel  = d.vel * std::exp(-4.f * FT);
            d.life -= FT;
        }
        std::erase_if(st.drops, [](const SDrop& d) { return d.life <= 0.f; });
    }

    // ------------------------------------------------------------ drawing ---

    void ensureMesh(SMonitorState& st) {
        if (st.vao)
            return;
        glGenVertexArrays(1, &st.vao);
        glGenBuffers(1, &st.posVbo);
        glGenBuffers(1, &st.uvVbo);
        glGenBuffers(1, &st.ebo);
        glBindVertexArray(st.vao);
        std::vector<float> grid;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                grid.insert(grid.end(), {-1.f + 2.f * i / (NX - 1), -1.f + 2.f * j / (NY - 1)});
        glBindBuffer(GL_ARRAY_BUFFER, st.uvVbo);
        glBufferData(GL_ARRAY_BUFFER, grid.size() * sizeof(float), grid.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, st.posVbo);
        glBufferData(GL_ARRAY_BUFFER, (size_t)NX * NY * 2 * sizeof(float), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        std::vector<uint16_t> idx;
        for (int j = 0; j + 1 < NY; ++j)
            for (int i = 0; i + 1 < NX; ++i) {
                const uint16_t A = j * NX + i, B = A + 1, C = A + NX, D = C + 1;
                idx.insert(idx.end(), {A, B, C, B, D, C});
            }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, st.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint16_t), idx.data(), GL_STATIC_DRAW);
        st.indexCount = (GLsizei)idx.size();

        glGenVertexArrays(1, &st.dropVao);
        glGenBuffers(1, &st.dropVbo);
        glBindVertexArray(st.dropVao);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.quad);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, st.dropVbo);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(4 * sizeof(float)));
        glVertexAttribDivisor(2, 1);

        glGenTextures(1, &st.waveId);
        glBindTexture(GL_TEXTURE_2D, st.waveId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void draw(SMonitorState& st, const SP<Render::IFramebuffer>& target) {
        if (!ensureGL())
            return;
        const auto TARGET = fbID(target);
        if (!TARGET || !st.snap || !ensureFB(st.scene, (int)target->m_size.x, (int)target->m_size.y, target->m_drmFormat, "hyprliquid-scene"))
            return;
        blitRect(target, st.scene, 0, 0, (int)target->m_size.x, (int)target->m_size.y); // what lies behind the liquid

        GLint prevProg = 0, prevVao = 0, prevBuf = 0, prevFb = 0, prevActive = 0, prevTex[3] = {}, vp[4];
        GLint srcRgb = 0, dstRgb = 0, srcA = 0, dstA = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFb);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActive);
        for (int u = 0; u < 3; ++u) {
            glActiveTexture(GL_TEXTURE0 + u);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex[u]);
        }
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &dstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);
        const GLboolean BLEND = glIsEnabled(GL_BLEND), SCISSOR = glIsEnabled(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, *TARGET);
        glViewport(0, 0, (GLsizei)target->m_size.x, (GLsizei)target->m_size.y);

        ensureMesh(st);
        // textures: 0 window, 1 scene, 2 waves
        glActiveTexture(GL_TEXTURE0);
        st.snap->getTexture()->bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glActiveTexture(GL_TEXTURE1);
        st.scene->getTexture()->bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, st.waveId);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, WX, WY, 0, GL_RED, GL_UNSIGNED_BYTE, st.waveTex.data());

        // the body
        glBindVertexArray(st.vao);
        glBindBuffer(GL_ARRAY_BUFFER, st.posVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, st.x.size() * sizeof(V2), st.x.data());
        glUseProgram(g_gl.liquid);
        const V2 E = st.bodyHalf + haloPx(st);
        glUniform2f(g_gl.lVFb, (float)target->m_size.x, (float)target->m_size.y);
        glUniform2f(g_gl.lExt, E.x, E.y);
        glUniform2f(g_gl.lBody, st.bodyHalf.x, st.bodyHalf.y);
        glUniform1f(g_gl.lCorner, std::min({st.corner, st.bodyHalf.x, st.bodyHalf.y}));
        glUniform1f(g_gl.lRim, (float)(RIM * st.scale));
        glUniform1f(g_gl.lHalo, (float)(HALO * st.scale));
        glUniform1f(g_gl.lPx, st.scale);
        glUniform2f(g_gl.lSlosh, st.slosh.x * st.glass, st.slosh.y * st.glass);
        glUniform1f(g_gl.lTime, secsSince(st.phaseStart));
        glUniform1f(g_gl.lGlass, st.glass);
        glUniform1i(g_gl.lWin, 0);
        glUniform1i(g_gl.lScene, 1);
        glUniform1i(g_gl.lWaves, 2);
        glDrawElements(GL_TRIANGLES, st.indexCount, GL_UNSIGNED_SHORT, nullptr);

        // droplets
        if (!st.drops.empty()) {
            std::vector<float> data;
            data.reserve(st.drops.size() * 6);
            for (const auto& d : st.drops) {
                const float K = d.life / d.maxLife;
                data.insert(data.end(), {d.pos.x, d.pos.y, d.r * (0.4f + 0.6f * K), K, d.uv.x, d.uv.y});
            }
            glBindVertexArray(st.dropVao);
            glBindBuffer(GL_ARRAY_BUFFER, st.dropVbo);
            glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STREAM_DRAW);
            glUseProgram(g_gl.drop);
            glUniform2f(g_gl.dVFb, (float)target->m_size.x, (float)target->m_size.y);
            glUniform1i(g_gl.dScene, 1);
            glUniform1i(g_gl.dWin, 0);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)st.drops.size());
            glDisable(GL_BLEND);
        }

        for (int u = 2; u >= 0; --u) {
            glActiveTexture(GL_TEXTURE0 + u);
            glBindTexture(GL_TEXTURE_2D, prevTex[u]);
        }
        glActiveTexture(prevActive);
        glUseProgram(prevProg);
        glBindVertexArray(prevVao);
        glBindBuffer(GL_ARRAY_BUFFER, prevBuf);
        glBindFramebuffer(GL_FRAMEBUFFER, prevFb);
        glViewport(vp[0], vp[1], vp[2], vp[3]);
        glBlendFuncSeparate(srcRgb, dstRgb, srcA, dstA);
        if (BLEND)
            glEnable(GL_BLEND);
        if (SCISSOR)
            glEnable(GL_SCISSOR_TEST);
    }

    // ------------------------------------------------------- pass element ---

    bool ownMonitorFrame() {
        const auto& rd = g_pHyprRenderer->m_renderData;
        if (g_pHyprRenderer->m_bRenderingSnapshot || rd.projectionType != Render::RPT_MONITOR)
            return false;
        if (rd.mainFB && rd.currentFB != rd.mainFB)
            return false;
        return !(rd.renderModif.enabled && !rd.renderModif.modifs.empty());
    }

    // The window being moved with the mouse, if it is on this monitor.
    PHLWINDOW movedWindow(const PHLMONITOR& mon) {
        if (!g_layoutManager)
            return nullptr;
        const auto& DC = g_layoutManager->dragController();
        if (!DC || DC->mode() != MBIND_MOVE)
            return nullptr;
        const auto T = DC->target();
        const auto W = T ? T->window() : nullptr;
        if (!W || !W->m_workspace || W->m_workspace != mon->m_activeWorkspace)
            return nullptr;
        return W;
    }

    class CElement : public IPassElement {
      public:
        std::vector<UP<IPassElement>> draw() override {
            const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
            const auto CUR = g_pHyprRenderer->m_renderData.currentFB;
            if (!MON || !CUR || CUR->m_size.x <= 0)
                return {};
            auto&      st = g_mon[MON->m_id];
            const auto DW = movedWindow(MON);

            if (st.phase == IDLE) {
                if (!DW || !begin(st, MON, DW, CUR))
                    return {};
                st.dragWin = DW;
            } else if (st.phase == HOLD && (!DW || DW != st.dragWin.lock())) {
                // dropped: pour into the window's new place
                const auto W = st.hiddenWin.lock();
                if (!W) {
                    st.phase = IDLE;
                    return {};
                }
                const CBox T  = glBox(CBox{W->positionAnimation()->goal(), W->sizeAnimation()->goal()}, MON);
                st.homeC      = {(float)(T.x + T.w * 0.5), (float)(T.y + T.h * 0.5)};
                st.startHalf  = st.bodyHalf;
                st.targetHalf = {(float)T.w * 0.5f, (float)T.h * 0.5f};
                st.phase      = RELEASE;
                st.phaseStart = Clock::now();
                stirWaves(st, {0.f, 0.f}, 0.8f, 6.f); // the splash of landing
            }
            if (st.phase == RELEASE && (secsSince(st.phaseStart) >= RELEASE_END || !st.hiddenWin.lock())) {
                showWindow(st);
                st.phase = IDLE;
                st.dragWin.reset();
                st.drops.clear();
                return {};
            }

            simulate(st, MON);
            ::draw(st, CUR);
            return {};
        }

        bool needsLiveBlur() override {
            return false;
        }
        bool needsPrecomputeBlur() override {
            return false;
        }
        const char* passName() override {
            return "CHyprliquidElement";
        }
        ePassElementType type() override {
            return EK_CUSTOM;
        }
        std::optional<CBox> boundingBox() override {
            return std::nullopt;
        }
        bool disableSimplification() override {
            return true;
        }
        bool undiscardable() override {
            return true;
        }
    };

    void onRenderStage(eRenderStage stage) {
        if (stage == RENDER_POST) {
            // keep frames coming (with full damage) while the liquid lives
            for (const auto& mon : State::monitorState()->monitors())
                if (mon && ((g_mon.contains(mon->m_id) && g_mon[mon->m_id].phase != IDLE) || movedWindow(mon)))
                    g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        if (stage != RENDER_POST_WINDOWS || !ownMonitorFrame())
            return;
        const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
        if (!MON)
            return;
        if (g_mon[MON->m_id].phase != IDLE || movedWindow(MON))
            g_pHyprRenderer->m_renderPass.add(makeUnique<CElement>());
    }
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    g_listeners.push_back(Event::bus()->m_events.render.stage.listen([](eRenderStage s) { onRenderStage(s); }));
    return {"hyprliquid", "A dragged window becomes a drop of liquid glass", "guizzan", "0.1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_listeners.clear();
    for (auto& [id, st] : g_mon) {
        showWindow(st); // never leave a window hidden
        if (st.vao)
            glDeleteVertexArrays(1, &st.vao);
        if (st.dropVao)
            glDeleteVertexArrays(1, &st.dropVao);
        for (GLuint b : {st.posVbo, st.uvVbo, st.ebo, st.dropVbo})
            if (b)
                glDeleteBuffers(1, &b);
        if (st.waveId)
            glDeleteTextures(1, &st.waveId);
    }
    g_mon.clear();
    if (g_gl.liquid)
        glDeleteProgram(g_gl.liquid);
    if (g_gl.drop)
        glDeleteProgram(g_gl.drop);
    if (g_gl.quad)
        glDeleteBuffers(1, &g_gl.quad);
    g_gl = {};
}
