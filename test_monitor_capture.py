"""
Test CreateForMonitor as an alternative to CreateForWindow.
If CreateForMonitor works, we can use monitor capture + crop = window capture for VRChat.
"""
import sys, ctypes, ctypes.wintypes, time
sys.path.insert(0, '.')

_combase = ctypes.WinDLL('combase.dll')
_d3d11   = ctypes.WinDLL('d3d11.dll')
hr = _combase.RoInitialize(1)
print(f'RoInit: 0x{hr&0xFFFFFFFF:08X}')

class _GUID(ctypes.Structure):
    _fields_=[('Data1',ctypes.c_ulong),('Data2',ctypes.c_ushort),('Data3',ctypes.c_ushort),('Data4',ctypes.c_uint8*8)]

# Setup argtypes
_combase.WindowsCreateString.restype  = ctypes.c_long
_combase.WindowsCreateString.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
_combase.WindowsDeleteString.restype  = ctypes.c_long
_combase.WindowsDeleteString.argtypes = [ctypes.c_void_p]
_combase.RoGetActivationFactory.restype  = ctypes.c_long
_combase.RoGetActivationFactory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
def _mk(s):
    s=s.strip('{}').upper().replace('-',''); g=_GUID()
    g.Data1=int(s[0:8],16); g.Data2=int(s[8:12],16); g.Data3=int(s[12:16],16)
    for i,b in enumerate(bytes.fromhex(s[16:])): g.Data4[i]=b; return g
def _hs(s): h=ctypes.c_void_p(); _combase.WindowsCreateString(s,len(s),ctypes.byref(h)); return h
def _qi(ptr, iid):
    if not ptr or not getattr(ptr,'value',None): return (-1, ctypes.c_void_p())
    out=ctypes.c_void_p(); vt=ctypes.cast(ptr,ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn=ctypes.cast(ctypes.cast(vt,ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.POINTER(_GUID),ctypes.POINTER(ctypes.c_void_p)))
    hr=fn(ptr,ctypes.byref(iid),ctypes.byref(out)); return (hr, out)

IID_IAF     = _mk('00000035-0000-0000-C000-000000000046')
IID_Ins     = _mk('AF86E2E0-B12D-4C6A-9C5A-D7AA65101E90')
IID_GCI     = _mk('79C3F95B-31F7-4EC2-A464-632EF5D30760')
IID_Interop = _mk('3628E81B-3CAC-4C60-B7F4-23CE0E0C3356')
IID_FPS1    = _mk('7784056A-67AA-4D53-AE54-1088D5A8CA21')
IID_DXG     = _mk('54EC77FA-1377-44E6-8C32-88FD5F44C84C')

class _SzI(ctypes.Structure): _fields_=[('Width',ctypes.c_int32),('Height',ctypes.c_int32)]

# D3D11 device
# D3D11 device - set argtypes properly
_d3d11.D3D11CreateDevice.restype  = ctypes.c_long
_d3d11.D3D11CreateDevice.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
]
_d3d11.CreateDirect3D11DeviceFromDXGIDevice.restype  = ctypes.c_long
_d3d11.CreateDirect3D11DeviceFromDXGIDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
dev=ctypes.c_void_p()
hr_d3d = _d3d11.D3D11CreateDevice(None,1,None,0x20,None,0,7,ctypes.byref(dev),None,None)
print(f'D3D11CreateDevice: 0x{hr_d3d&0xFFFFFFFF:08X}  dev=0x{dev.value or 0:016X}')
if hr_d3d != 0: raise SystemExit('D3D11 failed')
hr_dxgi_qi, dxgi = _qi(dev, IID_DXG)
print(f'QI IDXGIDevice: 0x{hr_dxgi_qi&0xFFFFFFFF:08X}  dxgi=0x{dxgi.value or 0:016X}')
if hr_dxgi_qi != 0: raise SystemExit('IDXGIDevice QI failed')
wd=ctypes.c_void_p()
hr_wd = _d3d11.CreateDirect3D11DeviceFromDXGIDevice(dxgi,ctypes.byref(wd))
print(f'CreateDirect3D11DeviceFromDXGIDevice: 0x{hr_wd&0xFFFFFFFF:08X}  wd=0x{wd.value or 0:016X}')
if hr_wd != 0: raise SystemExit('IDirect3DDevice failed')

# Get interop factory
h=_hs('Windows.Graphics.Capture.GraphicsCaptureItem'); fac=ctypes.c_void_p()
_combase.RoGetActivationFactory(h,ctypes.byref(IID_IAF),ctypes.byref(fac)); _combase.WindowsDeleteString(h)
_,interop=_qi(fac,IID_Interop)
print(f'Interop: 0x{interop.value:016X}')

# Get monitor HMONITOR for VRChat (the monitor it is on)
from capture_vrchat import find_vrchat_window, get_window_rect
hwnd, _ = find_vrchat_window()
print(f'VRChat hwnd={hwnd}')
l, t, r, b = get_window_rect(hwnd)
print(f'VRChat rect: {l},{t} -> {r},{b}  size={(r-l)}x{(b-t)}')

# Get monitor that VRChat is on
_MonitorFromWindow = ctypes.windll.user32.MonitorFromWindow
_MonitorFromWindow.restype = ctypes.c_void_p
_MonitorFromWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint32]
hmonitor = _MonitorFromWindow(ctypes.wintypes.HWND(hwnd), 2)  # MONITOR_DEFAULTTONEAREST=2
print(f'Monitor: 0x{hmonitor:016X}')

# Create capture item for monitor (vtable[4] of interop = CreateForMonitor)
monitor_item = ctypes.c_void_p()
vt_i = ctypes.cast(interop, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn4 = ctypes.cast(
    ctypes.cast(vt_i, ctypes.POINTER(ctypes.c_void_p))[4],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
)
hr_m = fn4(interop, ctypes.c_void_p(hmonitor), ctypes.byref(IID_Ins), ctypes.byref(monitor_item))
print(f'CreateForMonitor: hr=0x{hr_m&0xFFFFFFFF:08X}  item=0x{monitor_item.value or 0:016X}')

if hr_m == 0 and monitor_item.value:
    # QI to IGraphicsCaptureItem
    _,mon_gci = _qi(monitor_item, IID_GCI)
    print(f'Monitor QI IGraphicsCaptureItem: ptr=0x{mon_gci.value or 0:016X}')

    # Get size
    sz = _SzI()
    vt_m = ctypes.cast(mon_gci, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn7 = ctypes.cast(ctypes.cast(vt_m, ctypes.POINTER(ctypes.c_void_p))[7],
        ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_SzI)))
    fn7(mon_gci, ctypes.byref(sz))
    print(f'Monitor item size (vtable[7]): {sz.Width}x{sz.Height}')

    # Create pool
    h2=_hs('Windows.Graphics.Capture.Direct3D11CaptureFramePool'); fac2=ctypes.c_void_p()
    _combase.RoGetActivationFactory(h2,ctypes.byref(IID_IAF),ctypes.byref(fac2)); _combase.WindowsDeleteString(h2)
    _,fps1=_qi(fac2,IID_FPS1)
    # DispatcherQueue
    coremsg=ctypes.WinDLL('coremessaging.dll')
    class _DQO(ctypes.Structure): _fields_=[('dwSize',ctypes.c_uint32),('threadType',ctypes.c_uint32),('apartmentType',ctypes.c_uint32)]
    opts=_DQO(ctypes.sizeof(_DQO),2,0); dq=ctypes.c_void_p()
    coremsg.CreateDispatcherQueueController.argtypes=[_DQO,ctypes.POINTER(ctypes.c_void_p)]; coremsg.CreateDispatcherQueueController.restype=ctypes.c_long
    hr_dq = coremsg.CreateDispatcherQueueController(opts,ctypes.byref(dq))
    print(f'DispatcherQueueController: hr=0x{hr_dq&0xFFFFFFFF:08X}')

    pool=ctypes.c_void_p()
    vt_f=ctypes.cast(fps1,ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn6=ctypes.cast(ctypes.cast(vt_f,ctypes.POINTER(ctypes.c_void_p))[6],
        ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,ctypes.c_int32,_SzI,ctypes.POINTER(ctypes.c_void_p)))
    hr_p=fn6(fps1,wd,87,2,_SzI(sz.Width or 1920,sz.Height or 1080),ctypes.byref(pool))
    print(f'Create pool: hr=0x{hr_p&0xFFFFFFFF:08X}  pool=0x{pool.value or 0:016X}')

    # CreateCaptureSession
    session=ctypes.c_void_p()
    vt_p=ctypes.cast(pool,ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_cs=ctypes.cast(ctypes.cast(vt_p,ctypes.POINTER(ctypes.c_void_p))[6],
        ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)))
    hr_s=fn_cs(pool,mon_gci,ctypes.byref(session))
    print(f'CreateCaptureSession (monitor): hr=0x{hr_s&0xFFFFFFFF:08X}  session=0x{session.value or 0:016X}')

    if hr_s == 0 and session.value:
        print('*** MONITOR CAPTURE SESSION CREATED SUCCESSFULLY! ***')
        # Start session (vtable[6])
        vt_s=ctypes.cast(session,ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn_start=ctypes.cast(ctypes.cast(vt_s,ctypes.POINTER(ctypes.c_void_p))[6],
            ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p))
        hr_start=fn_start(session)
        print(f'StartCapture: hr=0x{hr_start&0xFFFFFFFF:08X}')
        # Wait for frame
        time.sleep(0.1)
        # TryGetNextFrame (pool vtable[8])
        frame=ctypes.c_void_p()
        fn_frame=ctypes.cast(ctypes.cast(vt_p,ctypes.POINTER(ctypes.c_void_p))[8],
            ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)))
        hr_frame=fn_frame(pool,ctypes.byref(frame))
        print(f'TryGetNextFrame: hr=0x{hr_frame&0xFFFFFFFF:08X}  frame=0x{frame.value or 0:016X}')
