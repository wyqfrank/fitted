# CLAUDE.md

## Frontend UI Consistency

This project should feel like one coherent product, not a collection of individually styled components.

The frontend stack is:

- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui for reusable UI primitives
- Radix UI only through shadcn/ui unless a missing primitive is genuinely needed
- lucide-react for icons
- `cn()` using `clsx` + `tailwind-merge` for conditional classes
- Framer Motion only for purposeful interaction or state transitions

Do not introduce another component library, icon library, styling system, or animation library without a clear technical reason.

---

## 1. Reuse Before Creating

Before building a new UI element:

1. Check whether an existing component already solves the problem.
2. Check whether a shadcn/ui primitive can be composed to solve it.
3. Extend an existing project component if the new behaviour is closely related.
4. Create a new component only when the abstraction will be reused or meaningfully simplifies the parent.

Do not create slightly different versions of the same:

- button
- card
- modal
- badge
- input
- tooltip
- dropdown
- status pill
- video container
- score display

Prefer variants over duplicated components.

---

## 2. Component Sources of Truth

Use these primitives consistently:

| UI need | Preferred implementation |
|---|---|
| Buttons | `Button` |
| Inputs | shadcn `Input` |
| Selects | shadcn `Select` |
| Dialogs | shadcn `Dialog` |
| Dropdowns | shadcn `DropdownMenu` |
| Tooltips | shadcn `Tooltip` |
| Toasts | project toast / shadcn-compatible toast |
| Badges / status | `Badge` or project `StatusBadge` |
| Icons | `lucide-react` |
| Loading | shared `Spinner` / skeleton primitives |
| Class merging | `cn()` |
| Motion | Framer Motion |

Do not hand-roll HTML controls when an established primitive exists unless the custom behaviour requires it.

---

## 3. Styling Rules

Use Tailwind utilities.

Avoid:

- inline `style={{ ... }}`
- CSS modules for ordinary component styling
- arbitrary one-off CSS files
- styled-components
- emotion
- duplicated hard-coded colour values

Prefer semantic design tokens through CSS variables and Tailwind.

Use the existing theme before adding new values.

If a repeated value appears across the UI, promote it to a token rather than copying it.

---

## 4. Design Tokens

The product uses a dark fashion-tech visual language.

Use semantic tokens rather than raw colours whenever practical:

```css
--background
--foreground
--card
--card-foreground
--muted
--muted-foreground
--border
--primary
--primary-foreground
--accent
--destructive
```

The visual hierarchy should generally be:

- near-black page background
- dark elevated surfaces
- subtle borders
- high-contrast primary text
- muted secondary text
- one strong accent treatment for important states
- restrained use of gradients and glow

Do not add random colours to differentiate components.

State colours should have consistent meanings across the app.

---

## 5. Spacing

Prefer the Tailwind spacing scale.

Use common spacing values repeatedly instead of arbitrary values.

Typical defaults:

- component internal gap: `gap-2` or `gap-3`
- related groups: `gap-4`
- card padding: `p-4` or `p-6`
- page section spacing: `gap-6` or `gap-8`
- large layout separation: `gap-8` or `gap-12`

Avoid classes such as:

```text
mt-[13px]
w-[417px]
gap-[19px]
```

unless the value is required for a genuinely precise layout.

---

## 6. Typography

Keep typography simple and hierarchical.

Prefer:

- large bold display text for battle states and FIT scores
- medium-weight headings
- normal body text
- muted small text for metadata and status

Do not invent different font sizes in every component.

Use Tailwind's standard scale wherever possible:

```text
text-xs
text-sm
text-base
text-lg
text-xl
text-2xl
text-4xl
text-6xl
```

Use tabular numbers for changing scores:

```tsx
className="tabular-nums"
```

This prevents the UI from shifting when numbers update.

---

## 7. Buttons

All clickable actions should use the shared `Button` component.

Use variants consistently:

- `default` — primary action
- `secondary` — secondary action
- `outline` — lower-emphasis action
- `ghost` — icon or lightweight control
- `destructive` — destructive action

Examples:

```tsx
<Button>Create Battle</Button>

<Button variant="outline">
  Copy Room Code
</Button>

<Button variant="ghost" size="icon">
  <Camera className="size-4" />
</Button>
```

Do not manually style `<button>` elements unless building a primitive.

---

## 8. Icons

Use `lucide-react` exclusively.

Example:

```tsx
import {
  Camera,
  Copy,
  LogOut,
  Wifi,
  WifiOff,
} from "lucide-react";
```

Default icon sizing:

```tsx
className="size-4"
```

For larger visual states:

```tsx
className="size-5"
className="size-6"
```

Do not use emojis as interface icons.

Do not mix icon packs.

---

## 9. Layout

Design for the two-laptop demo at common laptop widths. Mobile and narrow screens
are out of scope (see `docs/PRD.md` § 9).

Battle layout:

```text
Player 1 | VS | Player 2
```

Use CSS grid/flex classes rather than JavaScript window-size checks.

Prefer:

```tsx
className="grid gap-4 lg:grid-cols-[1fr_auto_1fr]"
```

Avoid fixed widths when a flexible layout constraint works better.

---

## 10. Video UI

All webcam feeds should use the same visual component.

Create/reuse a `PlayerCard` or `VideoPanel` rather than styling local and remote feeds independently.

Video defaults:

```tsx
className="aspect-video w-full rounded-2xl object-cover"
```

The surrounding container should own:

- player label
- connection state
- FIT score
- loading state
- disconnected state
- video overlay controls

The `<video>` element itself should remain simple.

Local and remote players should have visually symmetrical layouts unless there is a functional reason to differentiate them.

---

## 11. FIT Score UI

There should be one canonical score component.

Example API:

```tsx
<FittedScore
  score={82.4}
  confidence={0.91}
  state="live"
/>
```

Do not duplicate score formatting logic.

Score formatting should be centralised.

Use:

```tsx
score.toFixed(1)
```

unless product requirements change.

Use `tabular-nums`.

Score updates should be visually smooth but should not animate so aggressively that the number becomes difficult to read.

---

## 12. Connection and Status UI

Connection states should use a shared vocabulary:

```ts
type ConnectionState =
  | "idle"
  | "waiting"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";
```

Camera states should similarly be explicit:

```ts
type CameraState =
  | "idle"
  | "requesting"
  | "ready"
  | "denied"
  | "unavailable"
  | "error";
```

Do not scatter different strings for the same state throughout the UI.

Examples of canonical labels:

- `Waiting for opponent`
- `Connecting`
- `Camera ready`
- `Opponent connected`
- `Analysing fit`
- `Connection lost`
- `Camera unavailable`

---

## 13. Loading States

Every asynchronous UI action should have an intentional loading state.

Do not leave controls apparently clickable while an action is in progress.

Prefer:

```tsx
<Button disabled={isCreatingRoom}>
  {isCreatingRoom ? (
    <>
      <Loader2 className="size-4 animate-spin" />
      Creating…
    </>
  ) : (
    "Create Battle"
  )}
</Button>
```

Avoid full-screen spinners unless the entire application is genuinely blocked.

---

## 14. Empty and Error States

Errors should be actionable.

Bad:

```text
Something went wrong.
```

Better:

```text
Camera access was denied.
Enable camera permission in your browser and try again.
```

Error states should use the same card/layout language as the rest of the application rather than appearing as raw text.

Do not use `alert()` for product UI.

---

## 15. Animation

Animation should communicate state, not decorate everything.

Good uses:

- score transition
- opponent joined
- connection established
- winner reveal
- modal transition
- subtle hover feedback

Avoid:

- perpetual bouncing
- excessive gradients moving continuously
- large entrance animations on routine controls
- animation that delays interaction

Prefer CSS transitions for simple hover/focus effects.

Use Framer Motion only when stateful animation materially improves the interaction.

---

## 16. Accessibility

Every interactive element must be keyboard accessible.

Requirements:

- buttons must be actual buttons
- inputs must have labels
- icon-only buttons must have `aria-label`
- maintain visible focus states
- dialogs should trap focus through Radix/shadcn
- do not rely exclusively on colour to communicate state

Decorative icons should use:

```tsx
aria-hidden="true"
```

---

## 17. Client Components

Only add `"use client"` where browser APIs or React client state are required.

Examples that genuinely need client components:

- camera access
- WebRTC
- WebSocket signalling
- interactive room controls
- live score state

Do not convert an entire page tree to client components just because one child needs browser APIs.

Keep the client boundary as low as practical.

---

## 18. State Ownership

State should live at the lowest sensible shared owner.

Examples:

- camera stream → camera hook/component
- peer connection → WebRTC hook
- room state → battle room
- current FIT score → inference hook / battle state
- purely visual open/closed state → local component

Do not introduce a global state library unless state complexity genuinely requires it.

React state/context is preferred initially.

---

## 19. File Organisation

The frontend is rooted at `apps/web/`. Paths in the following frontend layout are
relative to that directory. Prefer:

```text
app/
  page.tsx
  room/
    [roomId]/
      page.tsx

components/
  battle/
    BattleRoom.tsx
    BattleResult.tsx
    PlayerCard.tsx
    FittedScore.tsx
  camera/
    CameraFeed.tsx
  ui/
    ...

hooks/
  useCamera.ts
  useWebRTC.ts
  useInference.ts

lib/
  capture-frame.ts
  rtc-config.ts
  scoring.ts
  utils.ts
```

The Python API is rooted at `services/inference/` and uses an installable `src/`
package. Keep online inference and API code there. Do not place notebooks, datasets,
checkpoints, or training-only dependencies in the deployable service package.

Do not put unrelated components into a giant generic `components` folder when a clear feature grouping exists.

---

## 20. Naming

Components:

```text
PascalCase
```

Hooks:

```text
useSomething
```

Utilities:

```text
camelCase
```

Files should generally use either the existing repository convention or:

```text
kebab-case.ts
kebab-case.tsx
```

Do not mix naming conventions arbitrarily.

---

## 21. TypeScript

Do not use `any` to make UI code compile.

Prefer explicit shared types.

Example:

```ts
type Player = {
  id: string;
  name: string;
  score: number | null;
  connected: boolean;
};
```

Use discriminated unions for complex UI states where appropriate.

---

## 22. Forms and Validation

For simple forms, standard React state is fine.

For non-trivial forms, use:

- React Hook Form
- Zod

Do not add a second form or validation system.

Room codes should be normalised consistently in one place.

---

## 23. Before Adding a Dependency

Before installing a frontend package, check:

1. Is the functionality already available in React/Next.js?
2. Is it already provided by shadcn/Radix?
3. Does the project already contain a dependency solving it?
4. Is the package worth increasing the dependency surface?

Do not install packages solely to avoid writing a few lines of straightforward code.

---

## 24. When Modifying Existing UI

Preserve the established visual system.

Before changing a component:

1. inspect adjacent components,
2. reuse their spacing and primitives,
3. preserve existing variants,
4. avoid introducing a new visual language for one screen.

If a requested design conflicts with an existing convention, adapt it to the project's existing system unless the request explicitly asks for a redesign.

---

## 25. Definition of Done for UI Work

Before considering frontend work complete, verify:

- no unnecessary new UI dependencies
- no duplicate primitives
- buttons use shared variants
- icons come from lucide-react
- colours use existing tokens
- loading states exist
- error states exist
- camera/WebRTC state is clearly communicated
- local and remote video panels are visually consistent
- TypeScript has no avoidable `any`
- no stale unused components/classes remain
- the page looks coherent at common laptop widths
- interactive states work with keyboard navigation

The priority is a polished, coherent hackathon product with a small and predictable UI system.

---

## 26. Documentation

Use each document for one purpose:

| Document | Holds |
|---|---|
| `docs/next-steps.md` | Outstanding work |
| `docs/PRD.md` | Product requirements |
| `docs/architecture.md` | Current architecture and key decisions |
| `docs/specs/` | Detailed interfaces, algorithms, schemas, and test plans |

Rules:

- Track work as `- [ ]` in `docs/next-steps.md` only. Do not reintroduce status
  checkboxes to the PRD or to the specs.
- Use `[x]` only after the stated result has been implemented and verified.
- Update the tracker in the same change that completes or removes an item.
- If an item leaves scope, move it to the out-of-scope section rather than checking it.
- When a decision is made, record it in `docs/architecture.md` with its rationale.
- Keep extensive interfaces, algorithms, schemas, experiments and test plans in
  `docs/specs/`, and link them rather than duplicating them.
