from geostat import integrity


def test_manifest_rejects_missing_products_and_invalid_json(tmp_path):
    products = {role: tmp_path / f"{role}.dat" for role in integrity.PRODUCTS}
    for role, path in products.items():
        path.write_bytes(role.encode("utf-8"))
    manifest = tmp_path / integrity.MANIFEST_NAME
    integrity.write_manifest(manifest, products)
    paths = {role: str(path) for role, path in products.items()}
    assert not integrity.verify_manifest(paths)
    paths[integrity.MANIFEST_ROLE] = str(manifest)
    assert integrity.verify_manifest(paths)
    paths["mask_cog"] = str(tmp_path / "other.dat")
    assert not integrity.verify_manifest(paths)
    paths["mask_cog"] = str(products["mask_cog"])
    products["mask_cog"].unlink()
    assert not integrity.verify_manifest(paths)
    products["mask_cog"].write_bytes(b"mask_cog")
    manifest.write_text('"not a manifest"', encoding="utf-8")
    assert not integrity.verify_manifest(paths)
