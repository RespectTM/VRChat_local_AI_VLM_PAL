import pygetwindow as gw

print('Checking VRChat windows in detail:')
windows = gw.getWindowsWithTitle('VRChat')
for i, window in enumerate(windows):
    print(f'Window {i}:')
    print(f'  Title: "{window.title}"')
    print(f'  Size: {window.width}x{window.height}')
    print(f'  Position: ({window.left}, {window.top}) to ({window.right}, {window.bottom})')
    print(f'  Visible: {window.visible}')
    print(f'  Minimized: {window.isMinimized}')
    print(f'  Active: {window.isActive}')
    print()

# Check all windows for VRChat-related
print('All VRChat-related windows:')
all_titles = gw.getAllTitles()
for title in sorted(all_titles):
    if 'vrchat' in title.lower() or 'vrc' in title.lower():
        windows = gw.getWindowsWithTitle(title)
        for window in windows:
            print(f'  "{title}" - Size: {window.width}x{window.height}, Visible: {window.visible}, Minimized: {window.isMinimized}')