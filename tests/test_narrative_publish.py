from pathlib import Path


def test_publish_bundle(tmp_path: Path):
    # Create fake run folder
    run_dir = tmp_path / "run"
    figs = run_dir / "figures"
    figs.mkdir(parents=True)
    meta = run_dir / "metadata.json"
    meta.write_text('{"protocol":"test","protocol_version":"0.0.1","inputs":["a.csv"]}')
    (run_dir / "report.txt").write_text("RQ report here")
    # fake figure
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(figs / "fig.png")
    plt.close()

    out_dir = tmp_path / "bundle"
    from foodspec.narrative import save_markdown_bundle

    save_markdown_bundle(run_dir, out_dir)
    assert (out_dir / "methods.md").exists()
    assert (out_dir / "figures.pdf").exists()
