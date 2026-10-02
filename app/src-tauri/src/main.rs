// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // webkit2gtk aborts inside Mesa (libgallium) during GL compositing on
    // RDNA4/AMD - the renderer process dies and takes the UI with it. Turn
    // off webkit's GL compositing path entirely; must be set before the
    // webview is created.
    std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
    llama_studio_lib::run()
}
