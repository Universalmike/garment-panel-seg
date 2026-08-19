# Design Note — Making the Fabric Fill Respect the Render's Shading

*Part 3. Reasoning only, no implementation.*

## The problem in one line

A flat-tiled swatch pasted onto a shaded 3D render reads as a sticker because it
throws away two things the render already knows: **how the light falls** (folds,
highlights, shadow in the drape) and **how the surface curves** (the fabric
should foreshorten and bend with the garment, not sit flat in screen space).

The good news is that the source is a **render**, not a photo. In a render we can
ask the 3D pipeline for extra passes cheaply. That changes the whole approach:
instead of trying to *infer* shading and geometry from pixels, we *read them off*
the render.

## The approach, from cheapest to best

I'd think of this as two separable jobs: **relighting** (make the fabric carry
the render's light and shadow) and **surface mapping** (make the pattern follow
the drape). They can be shipped independently.

### 1. Screen-space relight (cheapest, no new passes)

Treat the render's own brightness as a shading layer. Decompose the panel
roughly into *albedo × shading*: pull a normalised luminance/shading map from the
existing render (or from a dedicated ambient-occlusion / diffuse-lighting pass if
one exists), then **multiply** the tiled fabric by it instead of pasting the
fabric flat. Folds go darker, highlights stay bright, and a solid or lightly
textured fabric immediately stops looking like a sticker.

This is a small, deterministic change over the current flat tile. It fixes the
*lighting* half of the sticker problem but not the *geometry* half — the pattern
is still flat in screen space, so on a strong fold or a sleeve turning away it
will still look pasted-on.

### 2. UV-space fill + relight (the right answer for a render)

Because these are renders, the 3D pipeline can emit, per garment, a **UV pass**
(where each screen pixel lands in the garment's texture space) and a **normal
pass** (which way the surface faces). With those:

- Map the fabric **in texture space** using the UV pass, so the weave and any
  pattern follow the real surface — it foreshortens into folds and bends around
  the arm, seams line up where the panels meet.
- Relight the mapped fabric using the render's **lighting passes** (diffuse
  shading, ambient occlusion, and normals for any directional sheen).

This is the version that actually drapes, because it is doing what the renderer
does — just swapping the material's albedo for our fabric — rather than faking it
in 2D.

## What it needs that a flat tile does not

- **A shading signal.** At minimum the render's luminance; better, a dedicated
  AO / diffuse-lighting pass.
- **Geometry passes for the good version:** a UV pass and a normal pass per
  render. These are cheap to add in a 3D pipeline and are the thing a photo would
  never give you — the reason "it's a render" is a gift, not a constraint.
- **Fabric described as a tileable material,** not just a flat image: a base
  colour/pattern plus, ideally, how rough or shiny it is, so relighting looks
  right on satin vs. matte cotton.
- **An assumption that the panel mask and the passes are pixel-aligned** — they
  come from the same render, so they are, which keeps the whole thing
  deterministic.

## Where it breaks down

- **No geometry passes available.** If we only ever get the final RGB render, we
  are stuck at option 1 (screen-space relight). Patterns then distort on steep
  folds and where the surface turns away from camera.
- **Shiny / directional fabrics** (satin, sequins, leather). A single multiply
  can't reproduce specular highlights that move with the light; those need the
  normal pass and a proper shading model.
- **Sheer or semi-transparent fabrics**, where what's behind the cloth matters —
  a straight albedo swap ignores transmission.
- **Pattern scale and seams.** Keeping a check or stripe at a believable
  real-world scale, and matching it across the seam between two panels, is its
  own problem the mask alone doesn't solve.
- **Heavy self-occlusion** (a sleeve folded over the body) confuses screen-space
  shading transfer even when relighting is otherwise fine.

## What I'd ship first

**Option 1 — the screen-space multiply relight.** It is a small, deterministic
step beyond the current flat tile, needs no new render passes and no model, and
it removes the worst of the sticker look for the common case of solid and subtly
textured fabrics. It ships in the existing pipeline immediately.

In parallel I'd ask the render team for the **UV and normal passes**, because
that unlocks option 2 — the version that makes *patterned* fabrics actually drape
— and it's the correct long-term answer precisely because the source is a render.
Ship the relight now; wire up UV-space fill next.
