"""Point a parsed Forge env cfg at the recolored (real-print-color) asset USDs.

Usage (after parse_env_cfg and all other cfg mutations, before gym.make):
    from realcolor_override import apply_realcolor
    apply_realcolor(env_cfg)

Swaps only the dirname of each asset's spawn.usd_path to
~/forge_ts/real_assets/usd_realcolor/ (same filenames), so it is task-agnostic
and touches nothing else in the cfg. Raises if a recolored file is missing --
a silent fallback to original colors would corrupt the rehearsal eval the same
way a silent offset-injection failure would corrupt trials.jsonl.
"""
import os

RC_DIR = os.path.expanduser("~/forge_ts/real_assets/usd_realcolor")


def _swap(spawn_cfg, what):
    fname = os.path.basename(spawn_cfg.usd_path)
    new = os.path.join(RC_DIR, fname)
    if not os.path.isfile(new):
        raise RuntimeError(
            f"realcolor asset missing for {what}: {new} (run recolor_usd_assets.py first)"
        )
    spawn_cfg.usd_path = new
    return fname


def apply_realcolor(env_cfg):
    swapped = [
        _swap(env_cfg.task.fixed_asset.spawn, "fixed_asset"),
        _swap(env_cfg.task.held_asset.spawn, "held_asset"),
    ]
    for extra in ("small_gear_cfg", "large_gear_cfg"):
        cfg = getattr(env_cfg.task, extra, None)
        if cfg is not None:
            swapped.append(_swap(cfg.spawn, extra))
    print(f"[realcolor] swapped assets -> {RC_DIR}: {swapped}", flush=True)
    return env_cfg


def apply_table_color(rgb):
    """Bind a flat UsdPreviewSurface of the given rgb to every
    /World/envs/env_*/Table prim. strongerThanDescendants overrides the
    SeattleLabTable prototype's textured materials without de-instancing.
    Call AFTER gym.make (stage exists) and BEFORE env.reset().
    Raises if no Table prim is found -- a silent no-op would fake a
    light-table eval on a dark table."""
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, "/World/RealColorTableLooks/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/RealColorTableLooks/Mat/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    tables = [p for p in stage.Traverse()
              if p.GetName() == "Table" and str(p.GetPath()).startswith("/World/envs/")]
    if not tables:
        raise RuntimeError("apply_table_color: no /World/envs/*/Table prims found")

    # The SeattleLabTable uses NESTED instancing (de-instancing just the Table
    # root exposes no meshes -- verified 07-18). Level by level: find every
    # instance root inside the Table subtrees (proxy-aware traversal sees
    # inside instances), de-instance the ones that are real prims, rescan.
    # Outer roots are real on pass 1; inner roots become real once their
    # parent is de-instanced.
    for _ in range(8):  # nesting depth cap
        changed = False
        for t in tables:
            for prim in Usd.PrimRange(t, Usd.TraverseInstanceProxies()):
                if prim.IsInstance() and not prim.IsInstanceProxy():
                    rp = stage.GetPrimAtPath(prim.GetPath())
                    if rp:
                        rp.SetInstanceable(False)
                        changed = True
        if not changed:
            break

    n_meshes = 0
    for t in tables:
        for prim in Usd.PrimRange(t):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
                n_meshes += 1

    if n_meshes == 0:
        print("[realcolor] Table subtree dump (for diagnosis):", flush=True)
        for prim in Usd.PrimRange(tables[0], Usd.TraverseInstanceProxies()):
            print(f"  {prim.GetPath()} <{prim.GetTypeName()}> inst={prim.IsInstance()} proxy={prim.IsInstanceProxy()}", flush=True)
        raise RuntimeError(f"apply_table_color: tables={len(tables)} meshes=0 -- nothing recolored")
    print(f"[realcolor] table color {rgb}: {len(tables)} tables, {n_meshes} gprims bound", flush=True)
    return n_meshes
