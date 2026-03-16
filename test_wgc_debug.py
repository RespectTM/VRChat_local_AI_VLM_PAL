"""
Diagnostic script for WGC CreateCaptureSession debugging.
Tries different IIDs and approaches to identify the correct path.
"""
import sys, ctypes, ctypes.wintypes
sys.path.insert(0, '.')

_combase = ctypes.WinDLL('combase.dll')
_d3d11   = ctypes.WinDLL('d3d11.dll')
hr = _combase.RoInitialize(1)  # MTA
print(f'RoInit: 0x{hr&0xFFFFFFFF:08X}')

# ---- GUID helpers ----
class _GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', ctypes.c_ulong), ('Data2', ctypes.c_ushort),
        ('Data3', ctypes.c_ushort), ('Data4', ctypes.c_uint8 * 8)
    ]

def _mk(s):
    s = s.strip('{}').upper().replace('-', '')
    g = _GUID(); g.Data1=int(s[0:8],16); g.Data2=int(s[8:12],16); g.Data3=int(s[12:16],16)
    for i, b in enumerate(bytes.fromhex(s[16:])): g.Data4[i] = b
    return g

def _qi(ptr, iid):
    """QueryInterface — returns (hr, out_ptr)"""
    if not ptr or not ptr.value: return (-1, ctypes.c_void_p())
    out = ctypes.c_void_p()
    vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn = ctypes.cast(
        ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
    )
    hr = fn(ptr, ctypes.byref(iid), ctypes.byref(out))
    return (hr, out)

def _hs(s):
    h = ctypes.c_void_p()
    _combase.WindowsCreateString(s, len(s), ctypes.byref(h))
    return h

def _call_vt(ptr, idx, restype, argtypes, args):
    """Call vtable[idx] on ptr."""
    if not ptr or not ptr.value: return None
    vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_addr = ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))[idx]
    fn = ctypes.cast(fn_addr, ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes))
    return fn(ptr, *args)

# IIDs
IID_IAF     = _mk('00000035-0000-0000-C000-000000000046')
IID_Ins     = _mk('AF86E2E0-B12D-4C6A-9C5A-D7AA65101E90')
IID_Interop = _mk('3628E81B-3CAC-4C60-B7F4-23CE0E0C3356')
IID_GCI     = _mk('79C3F95B-31F7-4EC2-A464-632EF5D30760')  # actual on this machine
IID_FPS1    = _mk('7784056A-67AA-4D53-AE54-1088D5A8CA21')
IID_FPS2_SDK = _mk('589B103C-234E-4AC6-BD9B-F5A497AE6609')
IID_FPS2_PYD = _mk('589B103F-6BBC-5DF5-A991-02E28B3B66D5') # from pyd binary
IID_DXG     = _mk('54EC77FA-1377-44E6-8C32-88FD5F44C84C')

class _SzI(ctypes.Structure):
    _fields_ = [('Width', ctypes.c_int32), ('Height', ctypes.c_int32)]

# ---- D3D11 device ----
dev = ctypes.c_void_p()
hr = _d3d11.D3D11CreateDevice(None, 1, None, 0x20, None, 0, 7, ctypes.byref(dev), None, None)
print(f'D3D11CreateDevice: 0x{hr&0xFFFFFFFF:08X}  dev=0x{dev.value or 0:016X}')
if hr != 0: raise SystemExit('D3D11 failed')

hr_dxgi, dxgi = _qi(dev, IID_DXG)
print(f'QI IDXGIDevice: 0x{hr_dxgi&0xFFFFFFFF:08X}  dxgi=0x{dxgi.value or 0:016X}')

wd = ctypes.c_void_p()
hr = _d3d11.CreateDirect3D11DeviceFromDXGIDevice(dxgi, ctypes.byref(wd))
print(f'CreateDirect3D11DeviceFromDXGIDevice: 0x{hr&0xFFFFFFFF:08X}  wd=0x{wd.value or 0:016X}')
if hr != 0: raise SystemExit('IDirect3DDevice failed')

# ---- Find VRChat ----
from capture_vrchat import find_vrchat_window
hwnd, title = find_vrchat_window()
print(f'VRChat: hwnd={hwnd}  title={title}')
if not hwnd: raise SystemExit('No VRChat window')

# ---- Capture item ----
h = _hs('Windows.Graphics.Capture.GraphicsCaptureItem')
fac = ctypes.c_void_p()
_combase.RoGetActivationFactory(h, ctypes.byref(IID_IAF), ctypes.byref(fac))
_combase.WindowsDeleteString(h)
print(f'GCI Factory: 0x{fac.value or 0:016X}')

hr_i, interop = _qi(fac, IID_Interop)
print(f'QI IGraphicsCaptureItemInterop: 0x{hr_i&0xFFFFFFFF:08X}  ptr=0x{interop.value or 0:016X}')

item_ins = ctypes.c_void_p()
vt_i = ctypes.cast(interop, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn3 = ctypes.cast(
    ctypes.cast(vt_i, ctypes.POINTER(ctypes.c_void_p))[3],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.wintypes.HWND,
                       ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
)
hr = fn3(interop, ctypes.wintypes.HWND(hwnd), ctypes.byref(IID_Ins), ctypes.byref(item_ins))
print(f'CreateForWindow: 0x{hr&0xFFFFFFFF:08X}  item_ins=0x{item_ins.value or 0:016X}')

hr_gci, item = _qi(item_ins, IID_GCI)
print(f'QI IGraphicsCaptureItem: 0x{hr_gci&0xFFFFFFFF:08X}  item=0x{item.value or 0:016X}')
if hr_gci != 0: raise SystemExit('item QI failed')

# ---- Inspect item methods ----
vt_item = ctypes.cast(item, ctypes.POINTER(ctypes.c_void_p)).contents.value

# vtable[6] = get_DisplayName → returns HSTRING
rcn = ctypes.c_void_p()
fn_dn = ctypes.cast(ctypes.cast(vt_item, ctypes.POINTER(ctypes.c_void_p))[6],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
hr_dn = fn_dn(item, ctypes.byref(rcn))
if hr_dn == 0 and rcn.value:
    _PeekStr = _combase.WindowsGetStringRawBuffer
    _PeekStr.restype = ctypes.c_wchar_p; _PeekStr.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    print(f'item.DisplayName vtable[6] = {_PeekStr(rcn, None)!r}')
    _combase.WindowsDeleteString(rcn)
else:
    print(f'item.get_DisplayName vtable[6] hr=0x{hr_dn&0xFFFFFFFF:08X}')

# vtable[9] = get_Size → returns SizeInt32
sz = _SzI()
fn_sz = ctypes.cast(ctypes.cast(vt_item, ctypes.POINTER(ctypes.c_void_p))[9],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_SzI)))
hr_sz = fn_sz(item, ctypes.byref(sz))
print(f'item.get_Size vtable[9]: hr=0x{hr_sz&0xFFFFFFFF:08X}  size={sz.Width}x{sz.Height}')

# Raw: dump GetIids to see all IIDs (including possible different ones)
_CoTaskMemFree = ctypes.windll.ole32.CoTaskMemFree
_CoTaskMemFree.restype = None; _CoTaskMemFree.argtypes = [ctypes.c_void_p]
iid_count = ctypes.c_ulong(); iids_ptr = ctypes.c_void_p()
hr2 = _call_vt(item, 3, ctypes.c_long,
               [ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_void_p)],
               [ctypes.byref(iid_count), ctypes.byref(iids_ptr)])
print(f'item.GetIids: hr=0x{hr2&0xFFFFFFFF:08X}  count={iid_count.value}')
if hr2 == 0 and iids_ptr.value and iid_count.value > 0:
    raw = (ctypes.c_uint8 * (16 * iid_count.value)).from_address(iids_ptr.value)
    print(f'  raw IID bytes: {bytes(raw).hex()}')
    for i in range(iid_count.value):
        g = _GUID.from_buffer_copy(bytes(raw)[i*16:(i+1)*16])
        d4 = bytes(g.Data4)
        print(f'  [{i}] {g.Data1:08X}-{g.Data2:04X}-{g.Data3:04X}-{d4[:2].hex().upper()}-{d4[2:].hex().upper()}')
    _CoTaskMemFree(iids_ptr)

# Also check IUnknown identity
IID_IUnknown = _mk('00000000-0000-0000-C000-000000000046')
hr_unk, item_unk = _qi(item_ins, IID_IUnknown)
print(f'QI IUnknown: hr=0x{hr_unk&0xFFFFFFFF:08X}  same_ptr={item_unk.value == item_ins.value}')
IID_GCI_OLD = _mk('79B87781-7EAD-4E3B-BE60-4EBD3FE11B01')
hr_old, _ = _qi(item_ins, IID_GCI_OLD)
print(f'QI old IGraphicsCaptureItem (79B87781): hr=0x{hr_old&0xFFFFFFFF:08X}')


# ---- Frame pool factory ----
h2 = _hs('Windows.Graphics.Capture.Direct3D11CaptureFramePool')
fac2 = ctypes.c_void_p()
_combase.RoGetActivationFactory(h2, ctypes.byref(IID_IAF), ctypes.byref(fac2))
_combase.WindowsDeleteString(h2)
print(f'FPS Factory: 0x{fac2.value or 0:016X}')

# Try QI for various FPS2 IIDs + pyd IIDs from binary
pyd_iids = [
    ('FPS1',       '7784056A-67AA-4D53-AE54-1088D5A8CA21'),
    ('FPS2_SDK',   '589B103C-234E-4AC6-BD9B-F5A497AE6609'),
    ('FPS2_PYD',   '589B103F-6BBC-5DF5-A991-02E28B3B66D5'),
    ('PYD_167e0',  '743ED370-06EC-5040-A58A-901F0F757095'),
    ('PYD_167a0',  '37869CFA-2B48-5EBF-9AFB-DFFD805DEFDB'),
    ('PYD_167b0',  'FA50C623-38DA-4B32-ACF3-FA9734AD800E'),
]

print('\n--- Testing FPS statics IIDs ---')
for name, iid_str in pyd_iids:
    iid = _mk(iid_str)
    hr_q, statics = _qi(fac2, iid)
    if hr_q == 0 and statics.value:
        print(f'  {name} ({iid_str}): QI OK → 0x{statics.value:016X}')
        # Try calling vtable[6]
        pool = ctypes.c_void_p()
        vt = ctypes.cast(statics, ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn6 = ctypes.cast(
            ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))[6],
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, _SzI,
                               ctypes.POINTER(ctypes.c_void_p))
        )
        hr_p = fn6(statics, wd, 87, 2, _SzI(1294, 703), ctypes.byref(pool))
        print(f'    Create/FreeThreaded[6]: hr=0x{hr_p&0xFFFFFFFF:08X}  pool=0x{pool.value or 0:016X}')
        if hr_p == 0 and pool.value:
            session = ctypes.c_void_p()
            vt_p = ctypes.cast(pool, ctypes.POINTER(ctypes.c_void_p)).contents.value
            fn_cs = ctypes.cast(
                ctypes.cast(vt_p, ctypes.POINTER(ctypes.c_void_p))[6],
                ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p))
            )
            hr_s = fn_cs(pool, item, ctypes.byref(session))
            print(f'    CreateCaptureSession[6]: hr=0x{hr_s&0xFFFFFFFF:08X}  session=0x{session.value or 0:016X}')
            if hr_s == 0 and session.value:
                print(f'    *** SUCCESS! session=0x{session.value:016X} ***')
    else:
        print(f'  {name}: QI failed 0x{hr_q&0xFFFFFFFF:08X}')

# ---- Also inspect GraphicsCaptureAccess for permission check ----
print('\n--- Testing alternative item creation methods ---')

# Method 2: RoGetActivationFactory with IID_Interop directly (no IAF QI step)
h_direct = _hs('Windows.Graphics.Capture.GraphicsCaptureItem')
interop_direct = ctypes.c_void_p()
hr_d = _combase.RoGetActivationFactory(h_direct, ctypes.byref(IID_Interop), ctypes.byref(interop_direct))
_combase.WindowsDeleteString(h_direct)
print(f'RoGetActivationFactory(Interop directly): hr=0x{hr_d&0xFFFFFFFF:08X}  ptr=0x{interop_direct.value or 0:016X}')

if hr_d == 0 and interop_direct.value:
    # CreateForWindow requesting IID_GCI directly
    item2 = ctypes.c_void_p()
    vt_id = ctypes.cast(interop_direct, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn3d = ctypes.cast(
        ctypes.cast(vt_id, ctypes.POINTER(ctypes.c_void_p))[3],
        ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.wintypes.HWND,
                           ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
    )
    hr_fw2 = fn3d(interop_direct, ctypes.wintypes.HWND(hwnd), ctypes.byref(IID_GCI), ctypes.byref(item2))
    print(f'CreateForWindow(IID_GCI directly): hr=0x{hr_fw2&0xFFFFFFFF:08X}  item2=0x{item2.value or 0:016X}')

    if hr_fw2 == 0 and item2.value:
        # Try CreateCaptureSession with this item
        h3 = _hs('Windows.Graphics.Capture.Direct3D11CaptureFramePool')
        fac3 = ctypes.c_void_p()
        _combase.RoGetActivationFactory(h3, ctypes.byref(IID_IAF), ctypes.byref(fac3))
        _combase.WindowsDeleteString(h3)
        hr_fps, fps = _qi(fac3, IID_FPS1)
        pool3 = ctypes.c_void_p()
        vt3 = ctypes.cast(fps, ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn6 = ctypes.cast(ctypes.cast(vt3, ctypes.POINTER(ctypes.c_void_p))[6],
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, _SzI,
                               ctypes.POINTER(ctypes.c_void_p)))
        fn6(fps, wd, 87, 2, _SzI(1294, 703), ctypes.byref(pool3))
        session3 = ctypes.c_void_p()
        vt_p3 = ctypes.cast(pool3, ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn_cs3 = ctypes.cast(ctypes.cast(vt_p3, ctypes.POINTER(ctypes.c_void_p))[6],
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_void_p)))
        hr_s3 = fn_cs3(pool3, item2, ctypes.byref(session3))
        print(f'CreateCaptureSession (item from direct IID_GCI): hr=0x{hr_s3&0xFFFFFFFF:08X}  session=0x{session3.value or 0:016X}')

# Method 3: CreateForWindow using IID_IInspectable then try passing item_ins directly (no QI)
print('\nTrying item_ins (IInspectable*) directly without QI:')
h4 = _hs('Windows.Graphics.Capture.Direct3D11CaptureFramePool')
fac4 = ctypes.c_void_p()
_combase.RoGetActivationFactory(h4, ctypes.byref(IID_IAF), ctypes.byref(fac4))
_combase.WindowsDeleteString(h4)
_, fps4 = _qi(fac4, IID_FPS1)
pool4 = ctypes.c_void_p()
vt4 = ctypes.cast(fps4, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn6_4 = ctypes.cast(ctypes.cast(vt4, ctypes.POINTER(ctypes.c_void_p))[6],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, _SzI,
                       ctypes.POINTER(ctypes.c_void_p)))
fn6_4(fps4, wd, 87, 2, _SzI(1294, 703), ctypes.byref(pool4))
session4 = ctypes.c_void_p()
vt_p4 = ctypes.cast(pool4, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn_cs4 = ctypes.cast(ctypes.cast(vt_p4, ctypes.POINTER(ctypes.c_void_p))[6],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_void_p)))
hr_s4 = fn_cs4(pool4, item_ins, ctypes.byref(session4))
print(f'CreateCaptureSession(item_ins=IInspectable*): hr=0x{hr_s4&0xFFFFFFFF:08X}')

try:
    h3 = _hs('Windows.Graphics.Capture.GraphicsCaptureAccess')
    fac3 = ctypes.c_void_p()
    hr3 = _combase.RoGetActivationFactory(h3, ctypes.byref(IID_IAF), ctypes.byref(fac3))
    _combase.WindowsDeleteString(h3)
    print(f'  GraphicsCaptureAccess factory: hr=0x{hr3&0xFFFFFFFF:08X}')
except Exception as e:
    print(f'  GraphicsCaptureAccess: {e}')

# ---- Check IsSupported on GraphicsCaptureSession ----
print('\n--- GraphicsCaptureSession.IsSupported check ---')
try:
    GCS_IID_IAF_name = 'Windows.Graphics.Capture.GraphicsCaptureSession'
    h4 = _hs(GCS_IID_IAF_name)
    fac4 = ctypes.c_void_p()
    hr4 = _combase.RoGetActivationFactory(h4, ctypes.byref(IID_IAF), ctypes.byref(fac4))
    _combase.WindowsDeleteString(h4)
    print(f'  GCS factory: hr=0x{hr4&0xFFFFFFFF:08X}  fac=0x{fac4.value or 0:016X}')
    if hr4 == 0 and fac4.value:
        # Try calling vtable[6] (IsSupported?) with no args
        vt = ctypes.cast(fac4, ctypes.POINTER(ctypes.c_void_p)).contents.value
        for idx in range(6, 10):
            fn_addr = ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))[idx]
            # Try IsSupported: HRESULT IsSupported(bool* result)
            result = ctypes.c_bool(False)
            fn = ctypes.cast(fn_addr,
                ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)))
            try:
                hr5 = fn(fac4, ctypes.byref(result))
                print(f'  GCS factory vtable[{idx}]: hr=0x{hr5&0xFFFFFFFF:08X}  result={result.value}')
            except Exception as e2:
                print(f'  GCS factory vtable[{idx}]: {e2}')
except Exception as e:
    print(f'  GCS error: {e}')
