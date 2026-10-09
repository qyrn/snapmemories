import sys

datas = [
    ("snapmemories/templates", "snapmemories/templates"),
    ("snapmemories/static", "snapmemories/static"),
]

analysis = Analysis(
    ["snapmemories/__main__.py"],
    pathex=["."],
    datas=datas,
    excludes=["tkinter", "unittest", "pydoc", "doctest"],
    noarchive=False,
)
archive = PYZ(analysis.pure)

if sys.platform == "darwin":
    executable = EXE(
        archive,
        analysis.scripts,
        exclude_binaries=True,
        name="SnapMemories",
        console=False,
        upx=False,
    )
    collected = COLLECT(executable, analysis.binaries, analysis.datas, name="SnapMemories")
    app = BUNDLE(
        collected,
        name="SnapMemories.app",
        icon=None,
        bundle_identifier="com.qyrn.snapmemories",
    )
else:
    executable = EXE(
        archive,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        name="SnapMemories",
        icon="snapmemories/static/favicon.ico",
        console=False,
        upx=False,
    )
