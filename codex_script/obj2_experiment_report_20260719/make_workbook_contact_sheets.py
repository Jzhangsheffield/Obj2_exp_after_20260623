from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

folder = Path(__file__).resolve().parents[2] / "analysis" / "obj2_experiment_report_20260719" / "qa_workbook"
names = [
    "Overview", "Family Summary", "Top Models", "Selected Metrics",
    "Strict Last", "CL Configs", "FT Runs", "Class Recall",
    "Issues", "Diagnostics", "Checkpoint Pairs", "Module Summary",
    "Module Pairs", "Selected Modules", "Module Per Class", "Class Effects",
    "Effect Summary", "Bootstrap CIs", "Feature Diagnostics", "Reliability",
    "Metric Definitions", "Figure Guide", "Workbook Guide",
]
font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
for group_idx in range(0, len(names), 4):
    group = names[group_idx:group_idx+4]
    tiles = []
    for name in group:
        im = Image.open(folder / f"{name}.png").convert("RGB")
        scale = min(1.0, 1360 / im.width)
        im = im.resize((int(im.width*scale), int(im.height*scale)))
        tile = Image.new("RGB", (1400, im.height+55), "white")
        ImageDraw.Draw(tile).text((18,12), name, fill="#17365D", font=font)
        tile.paste(im, (20,50))
        tiles.append(tile)
    row_heights=[]
    for r in range(0,len(tiles),2): row_heights.append(max(t.height for t in tiles[r:r+2]))
    canvas=Image.new("RGB",(2800,sum(row_heights)),"#E5E7EB")
    y=0
    for r,h in enumerate(row_heights):
        for c,tile in enumerate(tiles[r*2:r*2+2]): canvas.paste(tile,(c*1400,y))
        y += h
    canvas.save(folder / f"contact_{group_idx//4+1}.png")
