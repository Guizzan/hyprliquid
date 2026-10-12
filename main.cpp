// hyprliquid — windows as liquid.
//
// Every change of a window is poured as liquid: opening (a drop swells into
// the window), closing (the window collapses into a drop and splashes),
// moving, resizing, fullscreen, layout changes (the window flows into its new
// shape) and workspace switches (the old desktop's windows flow out to one
// side, the new ones flow in from the other). Dragging a window with
// SUPER + mouse turns it into a drop that hangs from the cursor.
//
// Each liquid body is:
//   * a soft-body mesh (shape matching, Müller et al. 2005), pulled toward its
//     target (or the cursor), so it lags, stretches and settles;
//   * a small wave simulation on its surface, with sloshing driven by the
//     body's acceleration;
//   * rendered as a refractive liquid: the window's content seen through the surface
//     (refracted, with dispersion), a thick clear rim that bends what lies
//     behind it (red and blue bend differently), a meniscus pulling the
//     background in around it, specular light and a thin-film shimmer.
// Fast drags and closing windows shed droplets.
//
// Rendering: every frame the damaged part of the composited windows is kept
// in `last` and of the background (before windows) in `bg`. While bodies
// live, each frame the real windows under them are painted over with the
// background and the bodies are drawn from `last` (live content) or from a
// snapshot (closing windows, the dragged window). Hyprland's own window and
// workspace animations must be disabled.

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
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

static HANDLE PHANDLE = nullptr; // not inline: a GNU-unique symbol would pin the .so on unload

namespace {
    // shapes
    constexpr double DRAG_HALF   = 160.0; // a dragged window: half its longer side, logical px
    constexpr double HALO        = 30.0;  // meniscus around a body, logical px
    constexpr double RIM         = 22.0;  // clear glass rim, logical px
    constexpr double WIN_CORNER  = 10.0;  // window corner radius, logical px
    constexpr double EDGE        = 6.0;   // window boxes grown by this to take borders along, logical px
    constexpr int    NX          = 34;    // mesh vertices across (incl. the halo ring)
    constexpr int    NY          = 26;
    // timing (s)
    constexpr float DUR         = 0.50f; // open / close / move / workspace
    constexpr float GLASS_IN    = 0.12f;
    constexpr float GLASS_OUT   = 0.20f;
    constexpr float POUR_IN     = 0.34f; // drag: window -> drop
    constexpr float RELEASE_DUR = 0.55f; // drag: drop -> window
    // soft body
    constexpr float K_SHAPE = 900.f;  // pull toward the matched rest shape (1/s^2)
    constexpr float K_GRAB  = 2600.f; // a dragged drop follows the cursor
    constexpr float K_HOME  = 1100.f; // pull toward the target place
    constexpr float DAMP    = 30.f;   // 1/s (well damped: it settles instead of jiggling)
    constexpr float UPRIGHT = 0.55f;  // share of its tilt a body keeps per match
    constexpr float SUBSTEP = 1.f / 480.f;
    // surface waves
    constexpr int   WX = 96, WY = 64;
    constexpr float WAVE_STEP = 1.f / 120.f;
    // droplets
    constexpr size_t DROPS_MAX = 220;

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

    float smooth01(float t) {
        t = std::clamp(t, 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }
    float secsSince(Clock::time_point t) {
        return std::chrono::duration<float>(Clock::now() - t).count();
    }

    struct SDrop {
        V2    pos, vel;
        float r, life, maxLife;
        V2    uv; // where on its window it came from (its colour)
        int   body = -1; // texture source: index into the snapshot list at draw time (-1: none)
        SP<Render::IFramebuffer> tex;
    };

    enum eKind : uint8_t {
        DRAG,  // hangs from the cursor
        ENTER, // a window appears (open, or comes in with its workspace)
        MOVE,  // a window changes place / size
        EXIT   // a window goes away (close, or leaves with its workspace)
    };

    struct SBody {
        eKind                    kind = MOVE;
        uintptr_t                id   = 0;
        PHLWINDOWREF             win;
        bool                     closing = false; // EXIT by close: collapses and splashes
        bool                     hidden  = false; // the real window is hidden (drag)
        bool                     released = false;
        bool                     needSnap = false; // EXIT: copy its picture from `last` this frame
        CBox                     snapBox;          // GL px, for needSnap
        SP<Render::IFramebuffer> snap;             // its own picture (EXIT, DRAG); none: live from `last`
        CBox                     live;             // GL px: where the real window is now (content + painted over)
        Clock::time_point        start, lastStep, lastWave;
        float                    dur = DUR;
        // shape
        V2    startHalf, endHalf, bodyHalf, startC, homeC, grabN;
        float corner = 0, glass = 0, opacity = 1;
        // mesh
        std::vector<V2> x, v;
        V2              prevC, prevVel, vel, slosh, sloshV;
        // waves
        std::vector<float>   h, hv;
        std::vector<uint8_t> waveTex;
        float                nextRain = 0.3f;
        GLuint               waveId   = 0;
        bool                 burst    = false;
        bool                 dead     = false;
    };

    struct SWinBox {
        uintptr_t    id;
        PHLWINDOWREF win;
        CBox         box; // logical, grown by EDGE
    };

    struct SMonitorState {
        SP<Render::IFramebuffer> last, bg, scene;
        std::vector<std::unique_ptr<SBody>> bodies;
        std::vector<SDrop>                  drops;
        std::vector<SWinBox>                cached;
        WORKSPACEID                         cachedWs = WORKSPACE_INVALID;
        PHLWINDOWREF                        dragWin;
        float                               scale = 1.f;
        bool                                lastValid = false;
        std::mt19937                        rng{0x11};
    };

    std::unordered_map<MONITORID, SMonitorState> g_mon;
    std::vector<CHyprSignalListener>            g_listeners;
    std::vector<V2>                             g_grid; // rest position of each mesh vertex, [-1,1]

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
uniform sampler2D win;      // the window's picture
uniform sampler2D scene;    // the screen behind the liquid
uniform sampler2D waves;    // surface height, 0.5 = rest
uniform vec4  winRect;      // where in `win` the window is (uv x, y, w, h)
uniform vec2  fbSize;
uniform vec2  extHalf;      // rest half size incl. the halo
uniform vec2  bodyHalf;
uniform float corner;
uniform float rim;          // px
uniform float halo;         // px
uniform float px;           // monitor scale (1 logical px)
uniform vec2  slosh;        // surface tilt
uniform float time;
uniform float glass;        // 0 = plain window .. 1 = full liquid
uniform float opacity;      // fades the whole body (and its meniscus) out
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
    vec3  bare = texture(scene, spx / fbSize).rgb;

    if (d < 0.0) {
        // the meniscus: the background right next to the liquid is pulled in
        // toward it and darkens a little where the drop touches the glass
        float o = -d / (halo * ppx);
        if (o >= 1.0 || glass <= 0.001)
            discard;
        float prof = (1.0 - o) * (1.0 - o) * (1.0 - o);
        vec3  col  = sceneRGB(spx, n * 16.0 * px * prof * glass, 0.35);
        col       *= 1.0 - 0.22 * prof * glass;
        col += vec3(0.10) * glass * exp(-pow((o - 0.35) / 0.18, 2.0)) * max(dot(n, normalize(L.xy)), 0.0);
        frag = vec4(mix(bare, col, opacity), 1.0);
        return;
    }

    // surface: waves + slosh tilt
    vec2  wuv = clamp(p / (2.0 * bodyHalf) + 0.5, 0.0, 1.0);
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
    vec2 cuv  = 0.5 + (wuv - 0.5) * (1.0 - 0.10 * e2 * glass);
    vec2 refr = -N.xy * 0.045 * glass;
    vec2 a    = winRect.xy, s = winRect.zw;
    vec3 content = vec3(texture(win, a + clamp(cuv + refr * 1.35, 0.0, 1.0) * s).r,
                        texture(win, a + clamp(cuv + refr, 0.0, 1.0) * s).g,
                        texture(win, a + clamp(cuv + refr * 0.65, 0.0, 1.0) * s).b);

    // the clear rim: what lies behind, bent hard toward the inside
    vec3  behind = sceneRGB(spx, -n * (46.0 * px) * e2 - N.xy * 20.0 * px, 0.30);
    float clear  = smoothstep(0.30, 0.95, e) * 0.92 * glass;
    vec3  col    = mix(content, behind, clear);
    col *= 1.0 - 0.10 * glass * smoothstep(0.0, 0.35, e) * (1.0 - smoothstep(0.35, 0.9, e));

    // light
    vec3  H    = normalize(L + vec3(0.0, 0.0, 1.0));
    float nh   = max(dot(N, H), 0.0);
    float spec = pow(nh, 90.0) * 1.2 + pow(nh, 14.0) * 0.10;
    float fres = pow(1.0 - N.z, 1.6);
    col += vec3(spec + fres * 0.55) * glass;
    float line = exp(-pow((d - 1.6 * px * ppx) / (1.1 * px * ppx), 2.0));
    col += vec3(0.55) * line * (0.25 + 0.75 * max(dot(-n, normalize(L.xy)), 0.0)) * glass;
    vec3 film = 0.5 + 0.5 * cos(6.2831853 * (e * 1.4 + h * 0.6 + time * 0.12 + vec3(0.0, 0.33, 0.67)));
    col += film * e2 * e2 * 0.20 * glass;
    frag = vec4(mix(bare, col, opacity), 1.0);
}
)";

    const char* DROP_VERT = R"(#version 300 es
precision highp float;
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 drop;   // centre (GL px), radius, life left 0..1
layout(location = 2) in vec2 tint;   // uv in the tint texture
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
uniform sampler2D tintTex;
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
    col = mix(col, texture(tintTex, tintUv).rgb, 0.35);
    col *= 0.70 + 0.30 * N.z;
    vec3  L    = normalize(vec3(-0.45, -0.70, 0.55));
    float spec = pow(max(dot(N, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 60.0);
    col += vec3(spec * 1.1);
    float a = smoothstep(1.0, 1.0 - 1.5 / r, rho) * smoothstep(0.0, 0.25, life);
    frag = vec4(col * a, a);
}
)";

    struct SGL {
        GLuint liquid = 0, drop = 0, quad = 0, vao = 0, posVbo = 0, uvVbo = 0, ebo = 0, dropVao = 0, dropVbo = 0;
        GLsizei indexCount = 0;
        GLint   lFb = -1, lExt = -1, lBody = -1, lCorner = -1, lRim = -1, lHalo = -1, lPx = -1, lSlosh = -1, lTime = -1, lGlass = -1, lOpacity = -1, lWin = -1,
              lScene = -1, lWaves = -1, lRect = -1;
        GLint dFb = -1, dScene = -1, dTint = -1;
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

    // Programs and the shared mesh (call with a GL context, saves bindings itself).
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
        const auto U   = [](GLuint p, const char* n) { return glGetUniformLocation(p, n); };
        g_gl.lFb       = U(g_gl.liquid, "fbSize");
        g_gl.lExt      = U(g_gl.liquid, "extHalf");
        g_gl.lBody     = U(g_gl.liquid, "bodyHalf");
        g_gl.lCorner   = U(g_gl.liquid, "corner");
        g_gl.lRim      = U(g_gl.liquid, "rim");
        g_gl.lHalo     = U(g_gl.liquid, "halo");
        g_gl.lPx       = U(g_gl.liquid, "px");
        g_gl.lSlosh    = U(g_gl.liquid, "slosh");
        g_gl.lTime     = U(g_gl.liquid, "time");
        g_gl.lGlass    = U(g_gl.liquid, "glass");
        g_gl.lOpacity  = U(g_gl.liquid, "opacity");
        g_gl.lWin      = U(g_gl.liquid, "win");
        g_gl.lScene    = U(g_gl.liquid, "scene");
        g_gl.lWaves    = U(g_gl.liquid, "waves");
        g_gl.lRect     = U(g_gl.liquid, "winRect");
        g_gl.dFb       = U(g_gl.drop, "fbSize");
        g_gl.dScene    = U(g_gl.drop, "scene");
        g_gl.dTint     = U(g_gl.drop, "tintTex");

        GLint prevVao = 0, prevBuf = 0;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);

        glGenBuffers(1, &g_gl.quad);
        const float corners[] = {0, 0, 1, 0, 0, 1, 1, 1};
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.quad);
        glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);

        glGenVertexArrays(1, &g_gl.vao);
        glGenBuffers(1, &g_gl.posVbo);
        glGenBuffers(1, &g_gl.uvVbo);
        glGenBuffers(1, &g_gl.ebo);
        glBindVertexArray(g_gl.vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.uvVbo);
        glBufferData(GL_ARRAY_BUFFER, g_grid.size() * sizeof(V2), g_grid.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.posVbo);
        glBufferData(GL_ARRAY_BUFFER, g_grid.size() * sizeof(V2), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        std::vector<uint16_t> idx;
        for (int j = 0; j + 1 < NY; ++j)
            for (int i = 0; i + 1 < NX; ++i) {
                const uint16_t A = j * NX + i, B = A + 1, C = A + NX, D = C + 1;
                idx.insert(idx.end(), {A, B, C, B, D, C});
            }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_gl.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint16_t), idx.data(), GL_STATIC_DRAW);
        g_gl.indexCount = (GLsizei)idx.size();

        glGenVertexArrays(1, &g_gl.dropVao);
        glGenBuffers(1, &g_gl.dropVbo);
        glBindVertexArray(g_gl.dropVao);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.quad);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.dropVbo);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(4 * sizeof(float)));
        glVertexAttribDivisor(2, 1);

        glBindVertexArray(prevVao);
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

    bool ensureFB(SP<Render::IFramebuffer>& fb, const SP<Render::IFramebuffer>& like, const char* name) {
        return ensureFB(fb, (int)like->m_size.x, (int)like->m_size.y, like->m_drmFormat, name);
    }

    // Copy rect (x0,y0)-(x1,y1) of `from` to (dx,dy) in `to`. Leaves bindings and scissor as they were.
    void blitRect(const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to, int x0, int y0, int x1, int y1, int dx, int dy) {
        const auto A = fbID(from), B = fbID(to);
        if (!A || !B || x1 <= x0 || y1 <= y0)
            return;
        GLint prevRead = 0, prevDraw = 0;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
        const GLboolean SCISSOR = glIsEnabled(GL_SCISSOR_TEST);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, *A);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, *B);
        glBlitFramebuffer(x0, y0, x1, y1, dx, dy, dx + (x1 - x0), dy + (y1 - y0), GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prevRead);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDraw);
        if (SCISSOR)
            glEnable(GL_SCISSOR_TEST);
    }

    void blitAll(const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to) {
        blitRect(from, to, 0, 0, (int)from->m_size.x, (int)from->m_size.y, 0, 0);
    }

    // The render buffer only holds valid pixels inside this frame's damage:
    // copy just those rects into a persistent buffer.
    void blitDamage(const PHLMONITOR& mon, const SP<Render::IFramebuffer>& from, const SP<Render::IFramebuffer>& to) {
        CRegion region = g_pHyprRenderer->m_renderData.damage.copy();
        region.intersect(CBox{{}, mon->m_transformedSize});
        region.transform(Math::wlTransformToHyprutils(Math::invertTransform(mon->m_transform)), mon->m_transformedSize.x, mon->m_transformedSize.y);
        region.intersect(CBox{{}, from->m_size});
        for (const auto& r : region.getRects())
            blitRect(from, to, r.x1, r.y1, r.x2, r.y2, r.x1, r.y1);
    }

    CBox glBox(const CBox& b, const PHLMONITOR& mon) {
        const double S = mon->m_scale;
        return CBox{(b.x - mon->m_position.x) * S, (b.y - mon->m_position.y) * S, b.w * S, b.h * S}; // rows run top-down
    }

    V2 cursorGL(const PHLMONITOR& mon) {
        const Vector2D C = Pointer::mgr()->position();
        return {(float)((C.x - mon->m_position.x) * mon->m_scale), (float)((C.y - mon->m_position.y) * mon->m_scale)};
    }

    // ---------------------------------------------------------- the bodies ---

    V2 haloPx(float scale) {
        return {(float)(HALO * scale), (float)(HALO * scale)};
    }

    V2 restOf(const SBody& b, float scale, size_t i) {
        const V2 E = b.bodyHalf + haloPx(scale);
        return {g_grid[i].x * E.x, g_grid[i].y * E.y};
    }

    V2 centreOf(const SBody& b) {
        V2 c{};
        for (const auto& p : b.x)
            c = c + p;
        return c * (1.f / std::max<size_t>(b.x.size(), 1));
    }

    void stirWaves(SBody& b, V2 atN, float amount, float radius) {
        const float CX = (atN.x * 0.5f + 0.5f) * (WX - 1), CY = (atN.y * 0.5f + 0.5f) * (WY - 1);
        const int   R  = (int)std::ceil(radius * 2.f);
        for (int y = std::max(1, (int)CY - R); y <= std::min(WY - 2, (int)CY + R); ++y)
            for (int x = std::max(1, (int)CX - R); x <= std::min(WX - 2, (int)CX + R); ++x) {
                const float D2 = ((x - CX) * (x - CX) + (y - CY) * (y - CY)) / (radius * radius);
                b.hv[(size_t)y * WX + x] -= amount * std::exp(-D2);
            }
    }

    void stepWaves(SBody& b) {
        auto& h = b.h;
        auto& v = b.hv;
        for (int y = 1; y < WY - 1; ++y)
            for (int x = 1; x < WX - 1; ++x) {
                const size_t I = (size_t)y * WX + x;
                v[I] += (h[I - 1] + h[I + 1] + h[I - WX] + h[I + WX] - 4.f * h[I]) * 0.45f;
                v[I] *= 0.984f;
            }
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = std::clamp(h[i] + v[i], -1.f, 1.f);
        for (int x = 0; x < WX; ++x) {
            h[x]                         = h[x + WX];
            h[(size_t)(WY - 1) * WX + x] = h[(size_t)(WY - 2) * WX + x];
        }
        for (int y = 0; y < WY; ++y) {
            h[(size_t)y * WX]          = h[(size_t)y * WX + 1];
            h[(size_t)y * WX + WX - 1] = h[(size_t)y * WX + WX - 2];
        }
    }

    // A new body: mesh laid out at `startC` with half size `startHalf`.
    SBody& addBody(SMonitorState& st, eKind kind, uintptr_t id, V2 startC, V2 startHalf, V2 endC, V2 endHalf) {
        auto  B = std::make_unique<SBody>();
        auto& b = *B;
        b.kind      = kind;
        b.id        = id;
        b.startHalf = b.bodyHalf = startHalf;
        b.endHalf   = endHalf;
        b.startC    = startC;
        b.homeC     = endC;
        b.start = b.lastStep = b.lastWave = Clock::now();
        for (size_t i = 0; i < g_grid.size(); ++i) {
            b.x.push_back(startC + restOf(b, st.scale, i));
            b.v.push_back({});
        }
        b.prevC = startC;
        b.h.assign((size_t)WX * WY, 0.f);
        b.hv.assign((size_t)WX * WY, 0.f);
        b.waveTex.assign((size_t)WX * WY, 128);
        b.corner = (float)(WIN_CORNER * st.scale);
        st.bodies.push_back(std::move(B));
        return *st.bodies.back();
    }

    SBody* findBody(SMonitorState& st, uintptr_t id) {
        for (auto& b : st.bodies)
            if (b->id == id && !b->dead && b->kind != EXIT)
                return b.get();
        return nullptr;
    }

    float roundCorner(const SMonitorState& st, V2 half, float glass) {
        const float WINC = (float)(WIN_CORNER * st.scale);
        const float LIQ  = std::min(0.42f * std::min(half.x, half.y), (float)(70.0 * st.scale));
        return std::min(WINC + (std::max(LIQ, WINC) - WINC) * glass, std::min(half.x, half.y));
    }

    void stepBody(SBody& b, float scale, V2 cursor, float dt, float homeK) {
        const size_t N = b.x.size();
        const V2     c = centreOf(b);
        float        a00 = 0, a01 = 0, a10 = 0, a11 = 0;
        for (size_t i = 0; i < N; ++i) {
            const V2 D = b.x[i] - c, Q = restOf(b, scale, i);
            a00 += D.x * Q.x;
            a01 += D.x * Q.y;
            a10 += D.y * Q.x;
            a11 += D.y * Q.y;
        }
        const float TH = std::atan2(a10 - a01, a00 + a11) * UPRIGHT;
        const float CS = std::cos(TH), SN = std::sin(TH);
        const bool  GRAB = b.kind == DRAG && !b.released;
        const V2    GQ   = {b.grabN.x * b.bodyHalf.x, b.grabN.y * b.bodyHalf.y};
        const float SIG2 = std::pow(0.5f * std::min(b.bodyHalf.x, b.bodyHalf.y), 2.f) * 2.f;
        for (size_t i = 0; i < N; ++i) {
            const V2 Q    = restOf(b, scale, i);
            const V2 GOAL = c + V2{CS * Q.x - SN * Q.y, SN * Q.x + CS * Q.y};
            V2       acc  = (GOAL - b.x[i]) * K_SHAPE - b.v[i] * DAMP;
            if (GRAB) {
                const V2    DQ = Q - GQ;
                const float W  = std::exp(-(DQ.x * DQ.x + DQ.y * DQ.y) / SIG2);
                acc            = acc + ((cursor + DQ) - b.x[i]) * (K_GRAB * W);
            } else
                acc = acc + ((b.homeC + Q) - b.x[i]) * homeK;
            b.v[i] = b.v[i] + acc * dt;
        }
        for (size_t i = 0; i < N; ++i)
            b.x[i] = b.x[i] + b.v[i] * dt;
    }

    void emitDrops(SMonitorState& st, SBody& b, int count, float speedOut, const SP<Render::IFramebuffer>& tint, bool fromBack) {
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        const V2 C   = centreOf(b);
        const V2 DIR = len(b.vel) > 1.f ? b.vel * (1.f / len(b.vel)) : V2{0, 1};
        const float R = std::max(b.bodyHalf.x, b.bodyHalf.y);
        for (int k = 0; k < count && st.drops.size() < DROPS_MAX; ++k) {
            size_t best = 0;
            float  bestScore = 1e9f;
            for (size_t i = 0; i < b.x.size(); ++i) {
                const V2 Q = restOf(b, st.scale, i);
                // vertices on the body's edge
                if (std::abs(Q.x) > b.bodyHalf.x * 1.03f || std::abs(Q.y) > b.bodyHalf.y * 1.03f)
                    continue;
                if (std::abs(std::abs(Q.x) - b.bodyHalf.x) > 0.12f * b.bodyHalf.x && std::abs(std::abs(Q.y) - b.bodyHalf.y) > 0.12f * b.bodyHalf.y)
                    continue;
                const V2    D     = b.x[i] - C;
                const float SCORE = fromBack ? D.x * DIR.x + D.y * DIR.y + (uni(st.rng) - 0.5f) * R * 0.9f : uni(st.rng);
                if (SCORE < bestScore) {
                    bestScore = SCORE;
                    best      = i;
                }
            }
            const V2    Q    = restOf(b, st.scale, best);
            const V2    OUT  = (b.x[best] - C) * (1.f / std::max(len(b.x[best] - C), 1e-3f));
            const V2    PERP = {-DIR.y, DIR.x};
            const float LIFE = 0.35f + 0.45f * uni(st.rng);
            SDrop       d;
            d.pos     = b.x[best];
            d.vel     = fromBack ? b.vel * (0.25f + 0.3f * uni(st.rng)) + PERP * ((uni(st.rng) - 0.5f) * 260.f * st.scale)
                                 : OUT * (speedOut * (0.5f + uni(st.rng)) * st.scale) + V2{0.f, 120.f * st.scale * uni(st.rng)};
            d.r       = (2.5f + 6.f * uni(st.rng) * uni(st.rng)) * st.scale;
            d.life    = d.maxLife = LIFE;
            d.uv      = {Q.x / (2.f * b.bodyHalf.x) + 0.5f, Q.y / (2.f * b.bodyHalf.y) + 0.5f};
            d.tex     = tint;
            st.drops.push_back(d);
        }
    }

    // Advance one body by this frame. Returns false when it has finished.
    bool simulate(SMonitorState& st, SBody& b, const PHLMONITOR& mon) {
        const float T     = secsSince(b.start);
        const float FRAME = std::min(secsSince(b.lastStep), 1.f / 30.f);
        b.lastStep        = Clock::now();
        const V2 CUR      = cursorGL(mon);
        float    homeK    = K_HOME;

        switch (b.kind) {
            case DRAG: {
                if (!b.released) {
                    const float K = smooth01(T / POUR_IN);
                    b.bodyHalf    = b.startHalf + (b.endHalf - b.startHalf) * K;
                    b.glass       = smooth01(T / (POUR_IN * 0.7f));
                } else {
                    const float K = smooth01(T / (RELEASE_DUR * 0.65f));
                    b.bodyHalf    = b.startHalf + (b.endHalf - b.startHalf) * K;
                    b.glass       = 1.f - smooth01((T - RELEASE_DUR * 0.25f) / (RELEASE_DUR * 0.75f));
                    homeK         = K_HOME * smooth01(T / (RELEASE_DUR * 0.4f));
                    if (T >= RELEASE_DUR)
                        return false;
                }
                break;
            }
            default: {
                const float K = smooth01(T / (b.dur * 0.75f));
                b.bodyHalf    = b.startHalf + (b.endHalf - b.startHalf) * K;
                b.glass       = smooth01(T / GLASS_IN) * (1.f - smooth01((T - (b.dur - GLASS_OUT)) / GLASS_OUT));
                homeK         = K_HOME * (0.35f + 0.65f * smooth01(T / (b.dur * 0.5f)));
                if (b.kind == EXIT)
                    b.opacity = 1.f - smooth01((T - b.dur * 0.45f) / (b.dur * 0.55f));
                if (b.kind == EXIT && b.closing && !b.burst && T > b.dur * 0.45f) {
                    b.burst = true;
                    emitDrops(st, b, 26, 420.f, b.snap, false);
                }
                if (T >= b.dur)
                    return false;
            }
        }
        b.corner = roundCorner(st, b.bodyHalf, b.glass);

        float DT = FRAME;
        while (DT > 0.f) {
            const float H = std::min(DT, SUBSTEP);
            stepBody(b, st.scale, CUR, H, homeK);
            DT -= H;
        }

        // motion -> slosh
        const V2    C    = centreOf(b);
        const float STEP = std::max(FRAME, 1e-3f);
        const V2    VEL  = (C - b.prevC) * (1.f / STEP);
        b.prevC          = C;
        const V2 ACC     = (VEL - b.prevVel) * (1.f / STEP);
        b.prevVel        = VEL;
        b.vel            = VEL;
        const float R    = std::max(b.bodyHalf.x, b.bodyHalf.y);
        const V2    PUSH = ACC * (-0.00015f / std::max(R / 240.f, 0.3f));
        b.sloshV         = b.sloshV + (PUSH - b.slosh * 140.f - b.sloshV * 7.f) * STEP;
        b.slosh          = b.slosh + b.sloshV * STEP;
        b.slosh          = {std::clamp(b.slosh.x, -0.4f, 0.4f), std::clamp(b.slosh.y, -0.4f, 0.4f)};

        // waves
        const float WT = secsSince(b.lastWave);
        if (WT >= WAVE_STEP) {
            b.lastWave = Clock::now();
            std::uniform_real_distribution<float> uni(0.f, 1.f);
            const float SPEED = len(VEL) / (st.scale * 1000.f); // ~1 at 1000 logical px/s
            if (b.kind == DRAG && !b.released)
                stirWaves(b, b.grabN, std::min(SPEED, 2.f) * 0.07f, 3.5f);
            b.nextRain -= WT;
            if (b.nextRain <= 0.f) {
                stirWaves(b, {uni(st.rng) * 1.6f - 0.8f, uni(st.rng) * 1.6f - 0.8f}, 0.2f, 2.5f);
                b.nextRain = 0.25f + 0.5f * uni(st.rng);
            }
            for (int k = 0; k < std::min(4, (int)(WT / WAVE_STEP)); ++k)
                stepWaves(b);
            for (size_t i = 0; i < b.h.size(); ++i)
                b.waveTex[i] = (uint8_t)std::clamp((int)std::lround((b.h[i] * 0.5f + 0.5f) * 255.f), 0, 255);
            if (b.kind == DRAG && !b.released && T > POUR_IN && SPEED > 0.6f) {
                const int COUNT = (int)std::min(3.f, SPEED * 1.2f * uni(st.rng) + (uni(st.rng) < SPEED * 0.5f ? 1.f : 0.f));
                emitDrops(st, b, COUNT, 0.f, b.snap, true);
            }
        }
        return true;
    }

    // ---------------------------------------------------- window tracking ---

    std::vector<SWinBox> windowBoxes(const PHLWORKSPACE& ws, const PHLWINDOW& skip) {
        std::vector<SWinBox> out;
        for (const auto& w : Desktop::windowState()->windows()) {
            if (!w || !w->m_isMapped || w->isHidden() || w->m_workspace != ws || w == skip)
                continue;
            out.push_back({(uintptr_t)w.get(), w, CBox{w->positionAnimation()->goal(), w->sizeAnimation()->goal()}.expand(EDGE)});
        }
        return out;
    }

    const SWinBox* findBox(const std::vector<SWinBox>& v, uintptr_t id) {
        for (const auto& b : v)
            if (b.id == id)
                return &b;
        return nullptr;
    }

    bool sameBox(const CBox& a, const CBox& b) {
        return std::abs(a.x - b.x) < 0.5 && std::abs(a.y - b.y) < 0.5 && std::abs(a.w - b.w) < 0.5 && std::abs(a.h - b.h) < 0.5;
    }

    V2 centreOf(const CBox& b) {
        return {(float)(b.x + b.w * 0.5), (float)(b.y + b.h * 0.5)};
    }
    V2 halfOf(const CBox& b) {
        return {(float)b.w * 0.5f, (float)b.h * 0.5f};
    }

    // a window flows from `from` to `to` (or keeps flowing to a new target)
    void moveBody(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w, const CBox& fromL) {
        const CBox TO = glBox(w.box, mon);
        if (auto* b = findBody(st, w.id)) {
            if (b->kind == DRAG)
                return;
            b->startHalf = b->bodyHalf;
            b->endHalf   = halfOf(TO);
            b->homeC     = centreOf(TO);
            b->live      = TO;
            b->start     = Clock::now() - std::chrono::milliseconds((int)(GLASS_IN * 1000)); // stays liquid
            if (b->kind == ENTER)
                b->kind = MOVE;
            return;
        }
        const CBox FROM = glBox(fromL, mon);
        auto&      b    = addBody(st, MOVE, w.id, centreOf(FROM), halfOf(FROM), centreOf(TO), halfOf(TO));
        b.win           = w.win;
        b.live          = TO;
        stirWaves(b, {0.f, 0.f}, 0.5f, 8.f);
    }

    void enterBody(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w, float dirX) {
        const CBox TO = glBox(w.box, mon);
        const V2   C  = centreOf(TO), H = halfOf(TO);
        V2         sc = C, sh;
        if (dirX == 0.f) // opening: a drop swells into the window
            sh = {std::max(H.x * 0.12f, 18.f * st.scale), std::max(H.y * 0.12f, 18.f * st.scale)};
        else { // the workspace flows in from the side it lies on
            sc = C + V2{dirX * (float)mon->m_pixelSize.x * 0.45f, 0.f};
            sh = H * 0.55f;
        }
        auto& b = addBody(st, ENTER, w.id, sc, sh, C, H);
        b.win   = w.win;
        b.live  = TO;
        stirWaves(b, {0.f, 0.f}, 0.9f, 6.f);
    }

    void exitBody(SMonitorState& st, const PHLMONITOR& mon, const SWinBox& w, float dirX) {
        const CBox FROM = glBox(w.box, mon);
        const V2   C = centreOf(FROM), H = halfOf(FROM);
        V2         ec = C, eh;
        if (dirX == 0.f) // closing: collapses into a drop
            eh = {std::max(H.x * 0.10f, 14.f * st.scale), std::max(H.y * 0.10f, 14.f * st.scale)};
        else {
            ec = C - V2{dirX * (float)mon->m_pixelSize.x * 0.45f, 0.f};
            eh = H * 0.55f;
        }
        auto& b    = addBody(st, EXIT, w.id, C, H, ec, eh);
        b.closing  = dirX == 0.f;
        b.needSnap = true;
        b.snapBox  = FROM;
        stirWaves(b, {0.f, 0.f}, 0.6f, 6.f);
    }

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

    // Before the windows are drawn: compare the windows with last frame's.
    void watch(const PHLMONITOR& mon) {
        auto&      st = g_mon[mon->m_id];
        st.scale      = (float)mon->m_scale;
        const auto WS = mon->m_activeWorkspace;
        if (!WS || WS->m_isSpecialWorkspace)
            return;
        const auto DW    = movedWindow(mon);
        const auto BOXES = windowBoxes(WS, DW ? DW : PHLWINDOW{});

        if (st.cachedWs == WORKSPACE_INVALID || !st.lastValid) {
            st.cachedWs = WS->m_id;
            st.cached   = BOXES;
            return;
        }
        if (WS->m_id != st.cachedWs) {
            // workspace switch: the old windows flow out, the new ones flow in
            const float DIR = WS->m_id > st.cachedWs ? 1.f : -1.f;
            for (auto& b : st.bodies)
                if (b->kind != DRAG)
                    b->dead = true;
            for (const auto& w : st.cached)
                exitBody(st, mon, w, DIR);
            for (const auto& w : BOXES)
                enterBody(st, mon, w, DIR);
            st.cachedWs = WS->m_id;
            st.cached   = BOXES;
            g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        bool changed = false;
        for (const auto& w : BOXES) {
            const auto OLD = findBox(st.cached, w.id);
            if (!OLD) {
                // a window that was being dragged comes back into the list on drop: its body handles it
                if (auto* b = findBody(st, w.id); b && b->kind == DRAG)
                    continue;
                enterBody(st, mon, w, 0.f);
                changed = true;
            } else if (!sameBox(OLD->box, w.box)) {
                moveBody(st, mon, w, OLD->box);
                changed = true;
            }
        }
        for (const auto& w : st.cached)
            if (!findBox(BOXES, w.id)) {
                if (DW && w.id == (uintptr_t)DW.get())
                    continue; // picked up: the drag takes it
                if (auto* b = findBody(st, w.id))
                    b->dead = true;
                exitBody(st, mon, w, 0.f);
                changed = true;
            }
        st.cached = BOXES;
        if (changed)
            g_pHyprRenderer->damageMonitor(mon);
    }

    // ------------------------------------------------------------ drawing ---

    struct SGLState {
        GLint     prog = 0, vao = 0, buf = 0, fb = 0, active = 0, tex[3] = {}, vp[4] = {};
        GLint     srcRgb = 0, dstRgb = 0, srcA = 0, dstA = 0;
        GLboolean blend = 0, scissor = 0;
        void      save() {
            glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &buf);
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
            for (int u = 0; u < 3; ++u) {
                glActiveTexture(GL_TEXTURE0 + u);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex[u]);
            }
            glGetIntegerv(GL_VIEWPORT, vp);
            glGetIntegerv(GL_BLEND_SRC_RGB, &srcRgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &dstRgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcA);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &dstA);
            blend   = glIsEnabled(GL_BLEND);
            scissor = glIsEnabled(GL_SCISSOR_TEST);
        }
        void restore() {
            for (int u = 2; u >= 0; --u) {
                glActiveTexture(GL_TEXTURE0 + u);
                glBindTexture(GL_TEXTURE_2D, tex[u]);
            }
            glActiveTexture(active);
            glUseProgram(prog);
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, buf);
            glBindFramebuffer(GL_FRAMEBUFFER, fb);
            glViewport(vp[0], vp[1], vp[2], vp[3]);
            glBlendFuncSeparate(srcRgb, dstRgb, srcA, dstA);
            blend ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
            scissor ? glEnable(GL_SCISSOR_TEST) : glDisable(GL_SCISSOR_TEST);
        }
    };

    void linearClamp(const SP<Render::IFramebuffer>& fb) {
        fb->getTexture()->bind();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void drawBody(SMonitorState& st, SBody& b, const SP<Render::IFramebuffer>& target) {
        const SP<Render::IFramebuffer>& WIN = b.snap ? b.snap : st.last;
        if (!WIN)
            return;
        glActiveTexture(GL_TEXTURE0);
        linearClamp(WIN);
        if (!b.waveId) {
            glGenTextures(1, &b.waveId);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, b.waveId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, b.waveId);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, WX, WY, 0, GL_RED, GL_UNSIGNED_BYTE, b.waveTex.data());

        glBindVertexArray(g_gl.vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.posVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, b.x.size() * sizeof(V2), b.x.data());
        glUseProgram(g_gl.liquid);
        const V2    E  = b.bodyHalf + haloPx(st.scale);
        const float FW = (float)target->m_size.x, FH = (float)target->m_size.y;
        glUniform2f(g_gl.lFb, FW, FH);
        glUniform2f(g_gl.lExt, E.x, E.y);
        glUniform2f(g_gl.lBody, b.bodyHalf.x, b.bodyHalf.y);
        glUniform1f(g_gl.lCorner, b.corner);
        glUniform1f(g_gl.lRim, (float)(RIM * st.scale));
        glUniform1f(g_gl.lHalo, (float)(HALO * st.scale));
        glUniform1f(g_gl.lPx, st.scale);
        glUniform2f(g_gl.lSlosh, b.slosh.x * b.glass, b.slosh.y * b.glass);
        glUniform1f(g_gl.lTime, secsSince(b.start));
        glUniform1f(g_gl.lGlass, b.glass);
        glUniform1f(g_gl.lOpacity, b.opacity);
        if (b.snap)
            glUniform4f(g_gl.lRect, 0.f, 0.f, 1.f, 1.f);
        else
            glUniform4f(g_gl.lRect, (float)(b.live.x / FW), (float)(b.live.y / FH), (float)(b.live.w / FW), (float)(b.live.h / FH));
        glUniform1i(g_gl.lWin, 0);
        glUniform1i(g_gl.lScene, 1);
        glUniform1i(g_gl.lWaves, 2);
        glDrawElements(GL_TRIANGLES, g_gl.indexCount, GL_UNSIGNED_SHORT, nullptr);
    }

    void drawDrops(SMonitorState& st, const SP<Render::IFramebuffer>& target) {
        if (st.drops.empty())
            return;
        glUseProgram(g_gl.drop);
        glUniform2f(g_gl.dFb, (float)target->m_size.x, (float)target->m_size.y);
        glUniform1i(g_gl.dScene, 1);
        glUniform1i(g_gl.dTint, 0);
        glBindVertexArray(g_gl.dropVao);
        glBindBuffer(GL_ARRAY_BUFFER, g_gl.dropVbo);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        // one draw per tint texture (drops come from different windows)
        std::vector<float> data;
        for (size_t i = 0; i < st.drops.size();) {
            const auto TEX = st.drops[i].tex;
            data.clear();
            size_t j = i;
            for (; j < st.drops.size() && st.drops[j].tex == TEX; ++j) {
                const auto& d = st.drops[j];
                const float K = d.life / d.maxLife;
                data.insert(data.end(), {d.pos.x, d.pos.y, d.r * (0.4f + 0.6f * K), K, d.uv.x, d.uv.y});
            }
            if (TEX) {
                glActiveTexture(GL_TEXTURE0);
                linearClamp(TEX);
                glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STREAM_DRAW);
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)((j - i)));
            }
            i = j;
        }
        glDisable(GL_BLEND);
    }

    // ------------------------------------------------------- pass elements ---

    bool ownMonitorFrame() {
        const auto& rd = g_pHyprRenderer->m_renderData;
        if (g_pHyprRenderer->m_bRenderingSnapshot || rd.projectionType != Render::RPT_MONITOR)
            return false;
        if (rd.mainFB && rd.currentFB != rd.mainFB)
            return false;
        return !(rd.renderModif.enabled && !rd.renderModif.modifs.empty());
    }

    class CElement : public IPassElement {
      public:
        enum eStage : uint8_t {
            BG,
            POST
        };
        explicit CElement(eStage s) : m_stage(s) {}

        std::vector<UP<IPassElement>> draw() override {
            const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
            const auto CUR = g_pHyprRenderer->m_renderData.currentFB;
            if (!MON || !CUR || CUR->m_size.x <= 0)
                return {};
            auto& st = g_mon[MON->m_id];

            if (m_stage == BG) {
                // the background before any window, kept complete
                const bool FRESH = !st.bg || st.bg->m_size != CUR->m_size || st.bg->m_drmFormat != CUR->m_drmFormat;
                if (!ensureFB(st.bg, CUR, "hyprliquid-bg"))
                    return {};
                if (FRESH)
                    g_pHyprRenderer->damageMonitor(MON);
                blitDamage(MON, CUR, st.bg);
                return {};
            }

            // closing windows: their picture from the frame before
            const bool FRESH = !st.last || st.last->m_size != CUR->m_size || st.last->m_drmFormat != CUR->m_drmFormat;
            if (!ensureFB(st.last, CUR, "hyprliquid-last"))
                return {};
            if (FRESH) {
                g_pHyprRenderer->damageMonitor(MON);
                st.lastValid = false;
            }
            for (auto& b : st.bodies)
                if (b->needSnap) {
                    b->needSnap = false;
                    const CBox& B  = b->snapBox;
                    const int   X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
                    const int   X1 = std::min((int)CUR->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)CUR->m_size.y, (int)(B.y + B.h));
                    if (X1 - X0 < 4 || Y1 - Y0 < 4 || !ensureFB(b->snap, X1 - X0, Y1 - Y0, CUR->m_drmFormat, "hyprliquid-snap")) {
                        b->dead = true;
                        continue;
                    }
                    blitRect(st.last, b->snap, X0, Y0, X1, Y1, 0, 0);
                }
            blitDamage(MON, CUR, st.last); // now: this frame, as Hyprland drew it
            if (!FRESH)
                st.lastValid = true;

            // the dragged window
            const auto DW = movedWindow(MON);
            SBody*     drag = nullptr;
            for (auto& b : st.bodies)
                if (b->kind == DRAG && !b->dead)
                    drag = b.get();
            if (DW && !drag)
                startDrag(st, MON, DW, CUR);
            else if (drag && !drag->released && (!DW || (uintptr_t)DW.get() != drag->id))
                releaseDrag(st, *drag, MON);

            if (st.bodies.empty() && st.drops.empty())
                return {};
            if (!ensureGL())
                return {};

            // simulate; finished bodies hand back to the real window
            for (auto& b : st.bodies)
                if (!b->dead && !simulate(st, *b, MON))
                    b->dead = true;
            for (auto& b : st.bodies)
                if (b->dead) {
                    if (b->hidden)
                        showWindow(*b);
                    if (b->waveId) {
                        glDeleteTextures(1, &b->waveId);
                        b->waveId = 0;
                    }
                }
            std::erase_if(st.bodies, [](const auto& b) { return b->dead; });
            const float FT = 1.f / std::max(60.f, (float)MON->m_refreshRate);
            for (auto& d : st.drops) {
                d.pos = d.pos + d.vel * FT;
                d.vel = d.vel * std::exp(-4.f * FT) + V2{0.f, 600.f * st.scale * FT};
                d.life -= FT;
            }
            std::erase_if(st.drops, [](const SDrop& d) { return d.life <= 0.f; });
            if (st.bodies.empty() && st.drops.empty())
                return {};

            const auto TARGET = fbID(CUR);
            if (!TARGET || !st.bg)
                return {};
            SGLState gs;
            gs.save();
            glDisable(GL_BLEND);
            glDisable(GL_SCISSOR_TEST);

            // the real windows under live bodies give way to the background
            for (auto& b : st.bodies)
                if (!b->snap) {
                    const int X0 = std::max(0, (int)b->live.x), Y0 = std::max(0, (int)b->live.y);
                    const int X1 = std::min((int)CUR->m_size.x, (int)(b->live.x + b->live.w)), Y1 = std::min((int)CUR->m_size.y, (int)(b->live.y + b->live.h));
                    blitRect(st.bg, CUR, X0, Y0, X1, Y1, X0, Y0);
                }
            // what lies behind the liquid
            if (ensureFB(st.scene, CUR, "hyprliquid-scene"))
                blitAll(CUR, st.scene);

            glBindFramebuffer(GL_FRAMEBUFFER, *TARGET);
            glViewport(0, 0, (GLsizei)CUR->m_size.x, (GLsizei)CUR->m_size.y);
            glActiveTexture(GL_TEXTURE1);
            linearClamp(st.scene);
            // bodies leaving first, the dragged one last (on top)
            for (int pass = 0; pass < 3; ++pass)
                for (auto& b : st.bodies) {
                    const int P = b->kind == EXIT ? 0 : b->kind == DRAG ? 2 : 1;
                    if (P == pass)
                        drawBody(st, *b, CUR);
                }
            glActiveTexture(GL_TEXTURE1);
            linearClamp(st.scene);
            drawDrops(st, CUR);
            gs.restore();
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

      private:
        eStage m_stage;

        static void showWindow(SBody& b) {
            if (const auto W = b.win.lock())
                W->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(1.f);
            b.hidden = false;
        }

        static void startDrag(SMonitorState& st, const PHLMONITOR& mon, const PHLWINDOW& w, const SP<Render::IFramebuffer>& cur) {
            const CBox B  = glBox(CBox{w->positionAnimation()->value(), w->sizeAnimation()->value()}.expand(EDGE), mon);
            const int  X0 = std::max(0, (int)B.x), Y0 = std::max(0, (int)B.y);
            const int  X1 = std::min((int)cur->m_size.x, (int)(B.x + B.w)), Y1 = std::min((int)cur->m_size.y, (int)(B.y + B.h));
            SP<Render::IFramebuffer> snap;
            if (X1 - X0 < 8 || Y1 - Y0 < 8 || !ensureFB(snap, X1 - X0, Y1 - Y0, cur->m_drmFormat, "hyprliquid-drag"))
                return;
            blitRect(st.last, snap, X0, Y0, X1, Y1, 0, 0);
            const CBox  R  = {(double)X0, (double)Y0, (double)(X1 - X0), (double)(Y1 - Y0)};
            const V2    H  = halfOf(R), C = centreOf(R);
            const float BH = (float)(DRAG_HALF * st.scale);
            const float MX = std::max(H.x, H.y);
            const V2    TH = {std::max(BH * H.x / MX, 0.55f * BH), std::max(BH * H.y / MX, 0.55f * BH)};
            // any body the window had (it was flowing somewhere) gives way
            for (auto& b : st.bodies)
                if (b->id == (uintptr_t)w.get())
                    b->dead = true;
            auto& b  = addBody(st, DRAG, (uintptr_t)w.get(), C, H, C, TH);
            b.win    = w;
            b.snap   = snap;
            const V2 CUR = cursorGL(mon);
            b.grabN  = {std::clamp((CUR.x - C.x) / H.x, -0.8f, 0.8f), std::clamp((CUR.y - C.y) / H.y, -0.8f, 0.8f)};
            stirWaves(b, b.grabN, 0.9f, 5.f);
            w->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(0.f); // the liquid stands in for it
            b.hidden = true;
            st.dragWin = w;
        }

        static void releaseDrag(SMonitorState& st, SBody& b, const PHLMONITOR& mon) {
            const auto W = b.win.lock();
            if (!W) {
                b.dead = true;
                return;
            }
            const CBox T = glBox(CBox{W->positionAnimation()->goal(), W->sizeAnimation()->goal()}.expand(EDGE), mon);
            b.released   = true;
            b.startHalf  = b.bodyHalf;
            b.endHalf    = halfOf(T);
            b.homeC      = centreOf(T);
            b.start      = Clock::now();
            stirWaves(b, {0.f, 0.f}, 0.8f, 6.f); // the splash of landing
            st.dragWin.reset();
        }
    };

    void onRenderStage(eRenderStage stage) {
        if (stage == RENDER_POST) {
            // keep frames coming (with full damage) while the liquid lives
            for (const auto& mon : State::monitorState()->monitors())
                if (mon && g_mon.contains(mon->m_id) && (!g_mon[mon->m_id].bodies.empty() || !g_mon[mon->m_id].drops.empty() || movedWindow(mon)))
                    g_pHyprRenderer->damageMonitor(mon);
            return;
        }
        if (stage != RENDER_PRE_WINDOWS && stage != RENDER_POST_WINDOWS)
            return;
        if (!ownMonitorFrame())
            return;
        const auto MON = g_pHyprRenderer->m_renderData.pMonitor.lock();
        if (!MON)
            return;
        if (stage == RENDER_PRE_WINDOWS) {
            watch(MON);
            g_pHyprRenderer->m_renderPass.add(makeUnique<CElement>(CElement::BG));
        } else
            g_pHyprRenderer->m_renderPass.add(makeUnique<CElement>(CElement::POST));
    }
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            g_grid.push_back({-1.f + 2.f * i / (NX - 1), -1.f + 2.f * j / (NY - 1)});
    g_listeners.push_back(Event::bus()->m_events.render.stage.listen([](eRenderStage s) { onRenderStage(s); }));
    return {"hyprliquid", "Windows as liquid", "guizzan", "0.2"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_listeners.clear();
    for (auto& [id, st] : g_mon)
        for (auto& b : st.bodies) {
            if (b->hidden)
                if (const auto W = b->win.lock())
                    W->alpha(Desktop::View::WINDOW_ALPHA_LAYOUT)->setValueAndWarp(1.f); // never leave a window hidden
            if (b->waveId)
                glDeleteTextures(1, &b->waveId);
        }
    g_mon.clear();
    if (g_gl.liquid)
        glDeleteProgram(g_gl.liquid);
    if (g_gl.drop)
        glDeleteProgram(g_gl.drop);
    for (GLuint b : {g_gl.quad, g_gl.posVbo, g_gl.uvVbo, g_gl.ebo, g_gl.dropVbo})
        if (b)
            glDeleteBuffers(1, &b);
    for (GLuint v : {g_gl.vao, g_gl.dropVao})
        if (v)
            glDeleteVertexArrays(1, &v);
    g_gl = {};
}
