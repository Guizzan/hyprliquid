# hyprliquid — liquid glass for Hyprland

Windows that move like liquid glass. A Hyprland plugin that turns every window change into liquid. Each moving window is a soft body (shape matching) with a small wave simulation on its surface, rendered as Apple-style liquid glass: refraction, dispersion, a clear rim and a meniscus that pulls the background in.

**Demo and details: https://guizzan.github.io/hyprliquid/**

https://github.com/Guizzan/hyprliquid/raw/master/docs/demo.mp4

## What it does

- **Open / close** — A drop of glass swells into the new window; a closing window collapses into a drop and splashes into glass beads.
- **Move / resize / fullscreen / layout** — Every window that changes place flows into its new shape, lagging, stretching and settling.
- **Workspace switch** — The old desktop's windows flow out to one side while the new ones pour in from the other.
- **Drag** — SUPER + drag pours the window into a liquid drop hanging from the cursor: it sloshes, ripples and sheds droplets on fast moves, then pours into its new place.
- **Optics** — Content seen through the waving surface with dispersion, a thick clear rim bending what lies behind it, a meniscus, specular highlights and a thin-film shimmer.

## Install

Built and tested against **Hyprland 0.56** (Lua config). Plugins are tied to the exact Hyprland version — rebuild after every update.

```sh
make            # builds hyprliquid-dev.so
make install    # copies it to hyprliquid.so with a new inode (safe while Hyprland runs)
```

In your Lua config:

```lua
pcall(hl.plugin.load, os.getenv("HOME") .. "/path/to/hyprliquid/hyprliquid.so")
-- the plugin replaces Hyprland's own window and workspace animations
for _, leaf in ipairs({ "workspaces", "windowsIn", "windowsOut", "windowsMove", "fadeIn", "fadeOut" }) do
    hl.animation({ leaf = leaf, enabled = false })
end
```

Then log out and back in. Never hot-swap the plugin in a running session (unloading it can crash Hyprland), and run only one of these effect plugins at a time.

Try it in a nested session first: `Hyprland -c test/hyprland.lua`.

## See also

[hyprmorph](https://github.com/Guizzan/hyprmorph) (pixel morph) · [hyprhole](https://github.com/Guizzan/hyprhole) (black hole)
