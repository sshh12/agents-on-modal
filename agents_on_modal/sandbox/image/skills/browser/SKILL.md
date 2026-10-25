---
name: browser
description: Drive the headless Chromium in this sandbox to browse, scrape, fill forms, or test a web app. Use the browser_* MCP tools (Playwright) first; fall back to the agent-browser CLI or Playwright scripts. There is no display, so never expect a visible window.
---

# Browser

Headless Chromium 156 is installed at `/usr/bin/chromium` (no display). It is
already wired up for every tool below — do not run `playwright install` or pass
browser flags. To test a local server, bind it to `127.0.0.1:<port>` and browse
`http://127.0.0.1:<port>`.

## Primary: the `browser_*` MCP tools (Playwright)

Read the page as a text accessibility tree, then act on it by `ref`.

1. `browser_navigate` with `{"url": "..."}` to load a page.
2. `browser_snapshot` to read it. Only `browser_snapshot` returns the tree
   inline; after navigate/click you must call `browser_snapshot` again to see
   the new state. The tree looks like:
   `- textbox "Search" [ref=e3]`, `- button "Go" [ref=e4]`.
3. Act using the `eN` value as the **`target`** argument (the arg is `target`,
   not `ref`; optionally add `element` as a human description):
   - `browser_click {"target": "e4", "element": "Go button"}`
   - `browser_type {"target": "e3", "text": "playwright"}`
   - `browser_fill_form` to fill several fields at once
   - `browser_press_key {"key": "Enter"}`
   - `browser_select_option`, `browser_wait_for`, `browser_tabs`
4. Inspect with `browser_console_messages`, `browser_network_requests`, or
   `browser_evaluate` (run JS in the page).

Prefer snapshots over screenshots. `browser_take_screenshot` is configured to
omit the image for non-vision models and just save a file; only use it when a
pixel-level view is required.

## Fallback: the `agent-browser` CLI (for shell-driven work)

Keep one named session and avoid close/reopen churn (reopening right after a
close can error). Refs are short (`@e2`):

```sh
agent-browser open https://example.com
agent-browser snapshot -i          # interactive elements only, compact
agent-browser read                 # page text
agent-browser click @e2
agent-browser fill @e3 "some text"
agent-browser skills get core --full   # full, version-matched usage docs
```

## Fallback: scripts

Python `playwright` and Node `playwright`/`puppeteer-core` are installed. Launch
with the system browser — `chromium.launch(channel="chrome")` or
`chromium.launch(executable_path="/usr/bin/chromium")`. For a headed script use
`xvfb-run`. Never call `playwright install`.
