# Tanstarter → Tool Site Quickstart

Customize the Tanstarter template (a full SaaS boilerplate) into a simple
single-purpose SEO tool/game site. Proven on Marvel Rivals Randomizer.

## Files to Change

### 1. `package.json`
- Change `"name"` from `"mkfast-template"` to your project name

### 2. `src/config/website.ts`
- Disable all SaaS features by setting enable: false:
  - `auth.enable: false`, `auth.enableGoogleLogin: false`,
    `auth.enableCredentialLogin: false`, `auth.enableDeleteAccount: false`
  - `blog.enable: false`
  - `mail.enable: false`
  - `newsletter.enable: false`
  - `notification.enable: false`
  - `storage.enable: false`
  - `payment.enable: false`
- Update `metadata.name`, `metadata.title`, `metadata.description`
- Remove or empty `social`
- Remove locale message imports (the config used `m.site_name()` etc.)
- Set `ui.mode.enableSwitch: false` for simpler UX

### 3. `src/config/navbar-config.ts`
- Strip to minimal links: just the tool name and section anchors
- Replace locale imports (`m.nav_*()`) with static strings
- Example: `[{ title: 'Random Hero', href: '/', external: false }, ...]`

### 4. `src/config/footer-config.ts`
- Strip to minimal: Tool links + Legal links only
- Replace locale imports with static strings

### 5. `src/routes/__root.tsx`
- Remove: Analytics, CrispChat, LocaleSwitcher, UserButton, LoginWrapper imports
- Remove: auth-page and protected-page routing logic (only keep "is not found" check)
- Simplify head() meta tags — remove locale-specific OG tags
- Add a simple `Footer` component inline or in the root
- Remove script tags for Crisp, analytics scripts

### 6. `src/components/layout/navbar.tsx` (optional — rewrite for simplicity)
- Strip auth client, locale switcher, user button, mode switcher
- Use a simple `<nav>` with `<Link>` elements
- Keep mobile menu toggle for responsive UX
- Use `useScroll` hook for sticky header effect

### 7. `project.inlang/messages/en.json`
- Only keep the bare minimum keys the remaining components reference:
  `site_name`, `site_title`, `site_description`, `built_with_brand`,
  `footer_tagline`, `footer_rights_reserved`, `common_home`, etc.
- Update values away from Tanstarter defaults

### 8. `.env` from `.env.example`
```bash
cp -n .env.example .env
```

## Creating the Tool/Game Page

### Tab-based UI Pattern (proven on Marvel Rivals Randomizer)
```
┌─────────────────────────────────────┐
│  Tab Bar: Mode1 | Mode2 | Mode3     │
├─────────────────────────────────────┤
│  Content area with mode-specific UI │
│                                     │
│  - Submit button                    │
│  - Result display                   │
│  - Options/filters                  │
└─────────────────────────────────────┘
```

### Data Pattern for Game/Tool Sites
Create `src/lib/<tool-name>.ts` with:
- Interface/type definitions
- Static data array (characters, items, etc.)
- Utility functions: random selection, filtering, grouping
- Challenge/rule generation (optional)

### Homepage Component (`src/components/blocks/homepage.tsx`)
- Single tab-managed component with `useState` for `activeTab`
- Each tab renders a sub-component with its own state
- Common sub-components: `CharacterCard`, result display, FAQ accordion
- Use CSS gradients and backdrop blur for visual polish
- Include disclaimer/fan-made notice for IP-sensitive content
- Add Character Gallery section below the tool
- Add FAQ section with `<details>/<summary>` accordions

### SEO Structure
- H1: primary keyword in the page heading
- FAQ: structured Q&A for rich snippet eligibility
- Character/Item gallery: for long-tail keyword coverage
- Footer: legal links (Terms, Privacy, Cookie)

## Build & Deploy

```bash
cd /path/to/project
pnpm install        # First time
pnpm build          # Verify build succeeds
pnpm dev            # Dev server on port 3000
pnpm deploy         # Build + deploy to Cloudflare Workers
```
