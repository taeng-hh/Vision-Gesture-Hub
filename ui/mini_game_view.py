"""Retro space shooter dashboard with Korean sign controls."""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from ui.ksl_overlay import get_font


@lru_cache(maxsize=12)
def font(size):
    return get_font().font_variant(size=size)


def draw_galaga(game, camera, session, tracking):
    image = Image.new('RGB', (1280, 860), '#080c20')
    draw = ImageDraw.Draw(image)
    white, muted, mint = '#edf3ff', '#9aaaca', '#65edc2'

    def text(x, y, value, size=18, color=white):
        draw.text((x, y), str(value), font=font(size), fill=color)

    def card(box, fill='#121b32', outline='#2b4264'):
        draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline)

    def wrapped(value, x, y, width, size=16, max_lines=2):
        lines, line = [], ''
        for char in str(value):
            if draw.textlength(line+char, font=font(size)) > width:
                lines.append(line)
                line = ''
            line += char
        lines.append(line)
        for i, line in enumerate(lines[:max_lines]):
            text(x, y+i*(size+7), line, size, muted)

    text(32, 22, 'SIGN / GALAGA', 30)
    text(34, 66, '수화로 조종하는 레트로 우주 슈팅', 16, muted)
    for x, label, value in [(740, 'SCORE', f'{game.score:06d}'),
                            (920, 'LIVES', f'{game.lives} / 3'),
                            (1100, 'STAGE', f'{game.stage:02d}')]:
        card((x, 20, x+150, 94))
        text(x+16, 29, label, 12, muted)
        text(x+16, 50, value, 22, mint)

    card((28, 116, 858, 690), '#030715')
    text(50, 130, 'GALACTIC PATROL', 13, muted)
    text(580, 130, 'AUTO FIRE / '+('RAPID' if game.clock < game.boost_until else 'ON'), 13, mint)
    left, top, width, height = 48, 166, 790, 500

    def point(x, y):
        return left+x*width, top+y*height

    for i in range(85):
        x = (i*137 % 787)/790
        y = ((i*73 % 500)/500+game.clock*(.025+(i%3)*.012)) % 1
        sx, sy = point(x, y)
        draw.rectangle((sx, sy, sx+1, sy+1), fill=['#334365', '#7086a6', '#c0d9ff'][i%3])

    def sprite(x, y, rows, colors, scale=4):
        x, y = point(x, y)
        ox, oy = x-len(rows[0])*scale/2, y-len(rows)*scale/2
        for r, row in enumerate(rows):
            for c, pixel in enumerate(row):
                if pixel != '.':
                    draw.rectangle((ox+c*scale, oy+r*scale,
                                    ox+(c+1)*scale-1, oy+(r+1)*scale-1), fill=colors[pixel])

    enemy_sprite = ['..1...1..', '.121.121.', '111111111', '112111211', '..11111..', '.11...11.', '1.......1']
    for enemy in game.enemies:
        color = ['#ff778d', '#ffd166', '#a998ff'][enemy['row']]
        sprite(enemy['x'], enemy['y'], enemy_sprite, {'1': color, '2': '#071229'})
    for bullet in game.bullets:
        x, y = point(bullet['x'], bullet['y'])
        draw.rectangle((x-2, y-8, x+2, y+3), fill='#65edff')
    for bullet in game.enemy_bullets:
        x, y = point(bullet['x'], bullet['y'])
        draw.rectangle((x-3, y-3, x+3, y+7), fill='#ff8477')
    if game.clock >= game.invincible_until or int(game.clock*10) % 2 == 0:
        sprite(game.x, game.y, ['....1....', '....1....', '...111...', '1..121..1', '111121111', '111111111', '..1.3.1..', '....3....'],
               {'1': '#e8f6ff', '2': '#54c9ff', '3': '#ff9365'})

    overlay = None
    if game.state == 'ready': overlay = ('READY TO FLY?', '시작 수화 또는 Enter로 출격하세요. 탄환은 자동 발사됩니다.')
    elif game.state == 'over': overlay = ('GAME OVER', f'{game.score}점 획득! 시작 수화 또는 Enter로 다시 도전하세요.')
    elif session.recording: overlay = ('수화 인식 중', '동작을 마친 뒤 Space를 누르면 전투가 이어집니다.')
    elif game.state == 'paused': overlay = ('PAUSED', 'P를 누르면 계속합니다.')
    elif not tracking: overlay = ('카메라 추적 대기', '상반신과 손이 보이면 전투가 이어집니다.')
    if overlay:
        card((120, 340, 770, 470), '#142139', '#65edc2')
        text(146, 358, overlay[0], 28, mint)
        wrapped(overlay[1], 146, 409, 600, 17)

    card((882, 116, 1252, 436))
    text(902, 134, 'LIVE CAMERA', 13, muted)
    text(1090, 134, '녹화 중' if session.recording else ('추적 중' if tracking else '추적 대기'), 14, mint)
    preview = Image.fromarray(camera[:, ::-1, ::-1]).resize((330, 220))
    image.paste(preview, (902, 168))
    text(902, 401, '상반신과 양손이 보이게 해 주세요.', 14, muted)
    card((882, 452, 1252, 690))
    text(902, 468, '수화 컨트롤', 21)
    for i, (sign, action) in enumerate([('시작 · 달리다', '출격'), ('왼쪽 · 오른쪽', '좌우 이동'),
                                       ('위 · 아래', '상하 이동'), ('빠르다', '5초 연사')]):
        text(904, 509+i*33, sign, 17, muted)
        text(1095, 509+i*33, action, 17, mint)
    text(902, 651, '자동 발사 · 피격 후 2초 무적', 14, muted)
    card((28, 710, 1252, 798))
    text(50, 734, 'SPACE', 23, mint)
    text(185, 720, '수화 녹화 시작 / 완료', 18)
    wrapped(session.status, 185, 751, 1035, 15, 1)
    text(35, 818, 'P 일시정지    R 초기화    Q / ESC 나가기', 14, muted)
    text(655, 818, 'Enter 출격 · W/A/S/D 방향 이동 · F 연사', 14, muted)
    return np.asarray(image)[:, :, ::-1].copy()


# Keep callers of the former renderer working.
draw_runner = draw_galaga
