# Chrome AppleScript Interaction on macOS

Use this technique to extract content from the user's Chrome browser when:
- The user is logged into a web app (Similarweb, Web.Cafe, etc.)
- The Hermes browser tools are a separate session without credentials
- The task requires data from a page the user already has open

## Prerequisites

Enable "Allow JavaScript from Apple Events" in Chrome:
```
Chrome menu → View → Developer → Allow JavaScript from Apple Events
```
Without this, `execute ... javascript` will fail with error -1002.

## Core Patterns

### 1. Find tabs by URL pattern

```applescript
tell application "Google Chrome"
    set windowList to every window
    repeat with w in windowList
        set tabList to every tab of w
        repeat with t in tabList
            set tabURL to URL of t
            if tabURL contains "sim.3ue.co" then
                -- interact with this tab
            end if
        end repeat
    end repeat
end tell
```

### 2. Extract visible text

```applescript
set pageText to execute t javascript "document.body.innerText"
```

### 3. List all open tabs

```applescript
tell application "Google Chrome"
    set windowList to every window
    set output to ""
    repeat with w in windowList
        set tabList to every tab of w
        repeat with t in tabList
            set tabURL to URL of t
            set tabTitle to title of t
            set output to output & "⚡ " & tabTitle & return & "   " & tabURL & return
        end repeat
    end repeat
    return output
end tell
```

### 4. Click elements by text content (for SPAs)

```applescript
set result to execute t javascript "
(function() {
    var all = document.querySelectorAll('*');
    for(var i=0; i<all.length; i++) {
        var txt = all[i].textContent || '';
        if(txt.trim() === '着陆页') {
            all[i].click();
            return 'clicked: 着陆页';
        }
    }
    return 'not found';
})()
"
```

### 5. Navigate to a URL

```applescript
set URL of t to "https://example.com/page"
```
Note: SPA hash routing may NOT work with direct URL setting — the app may
not respond to hash-only navigation. Use DOM click events instead.

### 6. Input text into search fields

```applescript
set result to execute t javascript "
(function() {
    var input = document.querySelector('input[placeholder*=\"搜索\"]');
    if(input) {
        input.value = 'vercel.app';
        input.dispatchEvent(new Event('input', {bubbles: true}));
        var evt = new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true});
        input.dispatchEvent(evt);
        return 'submitted';
    }
    return 'not found';
})()
"
```

## Shell Integration (osascript)

Pass the entire AppleScript as a heredoc to avoid quote escaping hell:

```bash
osascript <<'ENDOSA'
tell application "Google Chrome"
    ...
end tell
ENDOSA
```

The `<<'ENDOSA'` syntax (single-quoted heredoc delimiter) tells bash not to
interpret $variables or backticks inside the block.

## Common Pitfalls

- **Empty responses** — `execute t javascript` returns the stringified
  return value. If the JS returns `undefined` or `null`, you get an empty
  string. Always return a meaningful string.
- **Timeout** — For slow SPAs, add `delay N` before `execute`.
- **SPA routing** — Hash-only URL changes may not trigger routing.
  Prefer DOM click simulation.
- **Quote nesting** — AppleScript strings use double-quotes. JS inside
  them also uses double-quotes. The single-quoted heredoc delimiter
  (`<<'ENDOSA'`) helps avoid bash interpretation issues.
