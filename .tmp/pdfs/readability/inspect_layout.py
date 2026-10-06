from pathlib import Path
from PIL import Image, ImageDraw

directory = Path(__file__).parent
pages = sorted(directory.glob("page-*.png"))
for start in range(0, len(pages), 6):
    canvas = Image.new("RGB", (1020, 1410), "white")
    draw = ImageDraw.Draw(canvas)
    for offset, path in enumerate(pages[start:start+6]):
        page = Image.open(path).convert("RGB")
        page.thumbnail((500, 665))
        x, y = (offset % 2)*510, (offset // 2)*470
        page.thumbnail((500, 445))
        canvas.paste(page, (x + (500-page.width)//2, y + 20))
        draw.text((x+15, y+3), path.stem, fill="black")
    canvas.save(directory / f"contact-{start//6+1}.png")
