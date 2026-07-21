from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

folder = Path(__file__).resolve().parents[2] / "analysis" / "obj2_experiment_report_20260719" / "qa_docx"
pages = sorted((folder / "pages").glob("page-*.png"))
font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
for start in range(0, len(pages), 4):
    group = pages[start:start+4]; tiles=[]
    for i,p in enumerate(group,start=start+1):
        im=Image.open(p).convert("RGB"); scale=850/im.width; im=im.resize((850,int(im.height*scale)))
        tile=Image.new("RGB",(880,im.height+45),"white"); ImageDraw.Draw(tile).text((15,8),f"Page {i}",fill="#17365D",font=font); tile.paste(im,(15,42)); tiles.append(tile)
    h=max(t.height for t in tiles); canvas=Image.new("RGB",(1760,h*2),"#D1D5DB")
    for j,tile in enumerate(tiles): canvas.paste(tile,((j%2)*880,(j//2)*h))
    canvas.save(folder/f"contact_{start//4+1}.png")
