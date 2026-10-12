-- Nested test session for hyprliquid (never used by the real session)
hl.config({
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true },
    general = { gaps_in = 8, gaps_out = 16, border_size = 2 },
})
for _, leaf in ipairs({ "workspaces", "windowsIn", "windowsOut", "windowsMove", "fadeIn", "fadeOut" }) do
    hl.animation({ leaf = leaf, enabled = false })
end
pcall(hl.plugin.load, os.getenv("HOME") .. "/Projects/hyprliquid/hyprliquid-dev.so")
hl.on("hyprland.start", function()
    hl.exec_cmd("kitty -o background=#c62828 --title A1", { workspace = "1 silent" })
    hl.exec_cmd("kitty -o background=#ef6c00 --title A2", { workspace = "1 silent" })
    hl.exec_cmd("kitty -o background=#1565c0 --title B1", { workspace = "2 silent" })
end)
