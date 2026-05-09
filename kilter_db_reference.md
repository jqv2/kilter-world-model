# Kilter Board Database Reference

Information from `notebooks/explore_kilter_db.ipynb`. Download kilter.db via `boardlib database kilter kilter.db`.

---

## Key IDs for the Board I'm using

| Entity | ID | Notes |
|---|---|---|
| Product | `product_id = 1` | Kilter Board Original |
| Layout | `layout_id = 1` | Kilter Board Original, `is_mirrored = 0` |
| Product Size | `product_size_id = 10` | 12×12 with kickboard |

## Board Dimensions

Board edges (from `product_sizes`): `edge_left=0, edge_right=144, edge_bottom=0, edge_top=156`

Actual hold position ranges (from `leds` table): **476 holds**, x=[4, 140], y=[4, 152]

Coordinate system: origin at bottom-left, x increases rightward, y increases upward. Kickboard holds have small y values (y=4, y=8). All other holds start higher.

## Placement Roles (product_id = 1)

| Role ID | Name | LED Color | Meaning |
|---|---|---|---|
| 12 | Start | Green (`00FF00`) | Starting holds |
| 13 | Middle | Cyan (`00FFFF`) | Hand holds |
| 14 | Finish | Magenta (`FF00FF`) | Finish holds |
| 15 | Foot Only | Orange (`FFA500`) | Foot-only holds |

## Route Encoding

Routes are stored in `climbs.frames` as a string: `p<placement_id>r<role_id>` repeated.

Example: `p1145r12p1216r13p1233r13...` → placement 1145 is a Start hold, 1216 is Middle, etc.

To decode a route into board coordinates:
```
climbs.frames -> parse p<id>r<role> pairs
-> placements (id -> hole_id, layout_id)
-> holes (hole_id -> x, y, name)
-> placement_roles (role_id -> start/middle/finish/foot)
```

## Hold Sets

| Set ID | Name | Notes |
|---|---|---|
| 1 | Bolt Ons | Main holds (`hsm=1`) |
| 20 | Screw Ons | Small footholds (`hsm=2`) |

## Route Statistics

- **315,357 climbs** on the Kilter Board Original layout
- `climb_stats` table has per-route, per-angle difficulty: `display_difficulty` maps to `difficulty_grades`
- Filter climbs at 30 degrees via: `JOIN climb_stats cs ON c.uuid = cs.climb_uuid AND cs.angle = 30`

## Difficulty Grade Mapping

| Difficulty | Boulder Grade |
|---|---|
| 10 | 4a/V0 |
| 11 | 4b/V0 |
| 12 | 4c/V0 |
| 13 | 5a/V1 |
| 14 | 5b/V1 |
| 15 | 5c/V2 |
| 16 | 6a/V3 |
| 17 | 6a+/V3 |
| 18 | 6b/V4 |
| 19 | 6b+/V4 |
| 20 | 6c/V5 |
| 21 | 6c+/V5 |
| 22 | 7a/V6 |
| 23 | 7a+/V7 |

Grades with `is_listed = 1` range from difficulty 10 (V0) to 33 (V16). Full mapping available in the `difficulty_grades` table.

## Source of Truth for Holds on the Board

Use the `leds` table filtered by `product_size_id = 10`, not the `placements` table filtered by edge boundaries. The layout has 692 placements total (covering all board sizes), but only 476 have LEDs wired on the 12×12 + kickboard size. The `leds` table gives the exact set of holds that physically exist on the board.

```sql
-- All holds on the board
SELECT h.id, h.x, h.y, h.name
FROM holes h
JOIN leds l ON l.hole_id = h.id
WHERE l.product_size_id = 10

-- Holds in a specific climb
SELECT p.id, h.x, h.y, h.name, pr.full_name
FROM placements p
JOIN holes h ON p.hole_id = h.id
LEFT JOIN placement_roles pr ON pr.id = <role_id>
WHERE p.id = <placement_id>
```