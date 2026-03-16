"""Find correct vtable index for get_Size on IGraphicsCaptureItem."""
import sys, ctypes, ctypes.wintypes
sys.path.insert(0, '.')

_combase = ctypes.WinDLL('combase.dll')
_d3d11   = ctypes.WinDLL('d3d11.dll')
hr = _combase.RoInitialize(1)  # MTA
print(f'RoInit: 0x{hr&0xFFFFFFFF:08X}')

class _GUID(ctypes.Structure):
    _fields_ = [('Data1',ctypes.c_ulong),('Data2',ctypes.c_ushort),
                ('Data3',ctypes.c_ushort),('Data4',ctypes.c_uint8*8)]
def _mk(s):
    s=s.strip('{}').upper().replace('-',''); g=_GUID()
    g.Data1=int(s[0:8],16); g.Data2=int(s[8:12],16); g.Data3=int(s[12:16],16)
    for i,b in enumerate(bytes.fromhex(s[16:])): g.Data4[i]=b
    return g
def _hs(s):
    h=ctypes.c_void_p(); _combase.WindowsCreateString(s,len(s),ctypes.byref(h)); return h
def _qi(ptr, iid):
    if not ptr or not getattr(ptr, 'value', None): return (-1, ctypes.c_void_p())
    out=ctypes.c_void_p()
    vt=ctypes.cast(ptr,ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn=ctypes.cast(ctypes.cast(vt,ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.POINTER(_GUID),ctypes.POINTER(ctypes.c_void_p)))
    hr=fn(ptr,ctypes.byref(iid),ctypes.byref(out))
    return (hr, out)

IID_IAF     = _mk('00000035-0000-0000-C000-000000000046')
IID_Ins     = _mk('AF86E2E0-B12D-4C6A-9C5A-D7AA65101E90')
IID_Interop = _mk('3628E81B-3CAC-4C60-B7F4-23CE0E0C3356')
IID_GCI     = _mk('79C3F95B-31F7-4EC2-A464-632EF5D30760')

class _SzI(ctypes.Structure):
    _fields_ = [('Width', ctypes.c_int32), ('Height', ctypes.c_int32)]

# ---- Create item ----
h = _hs('Windows.Graphics.Capture.GraphicsCaptureItem')
fac = ctypes.c_void_p()
hr_fac = _combase.RoGetActivationFactory(h, ctypes.byref(IID_IAF), ctypes.byref(fac))
_combase.WindowsDeleteString(h)
print(f'RoGetActivationFactory: 0x{hr_fac&0xFFFFFFFF:08X}  fac=0x{fac.value or 0:016X}')
if hr_fac != 0: raise SystemExit('factory failed')

hr_i, interop = _qi(fac, IID_Interop)
print(f'QI IGraphicsCaptureItemInterop: 0x{hr_i&0xFFFFFFFF:08X}  ptr=0x{interop.value or 0:016X}')
if hr_i != 0: raise SystemExit('interop QI failed')

from capture_vrchat import find_vrchat_window
hwnd, _ = find_vrchat_window()
print(f'hwnd = {hwnd}')

item_ins = ctypes.c_void_p()
vt_i = ctypes.cast(interop, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn3 = ctypes.cast(
    ctypes.cast(vt_i, ctypes.POINTER(ctypes.c_void_p))[3],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.wintypes.HWND,
                       ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p))
)
hr3 = fn3(interop, ctypes.wintypes.HWND(hwnd), ctypes.byref(IID_Ins), ctypes.byref(item_ins))
print(f'CreateForWindow: 0x{hr3&0xFFFFFFFF:08X}  item_ins=0x{item_ins.value or 0:016X}')
if hr3 != 0: raise SystemExit('CreateForWindow failed')

hr_g, item = _qi(item_ins, IID_GCI)
print(f'QI IGraphicsCaptureItem: 0x{hr_g&0xFFFFFFFF:08X}  item=0x{item.value or 0:016X}')
if hr_g != 0: raise SystemExit('item QI failed')

vt = ctypes.cast(item, ctypes.POINTER(ctypes.c_void_p)).contents.value

# Try vtable[6..12] treating each as get_Size (HRESULT(SizeInt32*))
print('\n--- Trying vtable indices as get_Size ---')
for idx in range(6, 13):
    sz = _SzI()
    fn_addr = ctypes.cast(vt, ctypes.POINTER(ctypes.c_void_p))[idx]
    fn = ctypes.cast(fn_addr,
                     ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_SzI)))
    try:
        hr_v = fn(item, ctypes.byref(sz))
        print(f'  vtable[{idx}]: hr=0x{hr_v&0xFFFFFFFF:08X}  size={sz.Width}x{sz.Height}  fn=0x{fn_addr:016X}')
    except Exception as e:
        print(f'  vtable[{idx}]: exception {type(e).__name__}: {e}')

# Also: get the item via IInspectable (not QI'd) and check its size at vtable[9]
print('\n--- IInspectable vtable size -----')
vt_ins = ctypes.cast(item_ins, ctypes.POINTER(ctypes.c_void_p)).contents.value
sz2 = _SzI()
fn_9 = ctypes.cast(ctypes.cast(vt_ins, ctypes.POINTER(ctypes.c_void_p))[9],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_SzI)))
hr_9 = fn_9(item_ins, ctypes.byref(sz2))
print(f'  item_ins vtable[9]: hr=0x{hr_9&0xFFFFFFFF:08X}  size={sz2.Width}x{sz2.Height}')
