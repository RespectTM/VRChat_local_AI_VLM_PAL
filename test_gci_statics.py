"""
Find IGraphicsCaptureItemStatics / IGraphicsCaptureItemStatics2 IID
and try TryCreateFromWindowId on Windows 11.
"""
import sys, ctypes, ctypes.wintypes
sys.path.insert(0, '.')

_combase = ctypes.WinDLL('combase.dll')
hr = _combase.RoInitialize(1)
print(f'RoInit: 0x{hr&0xFFFFFFFF:08X}')

class _GUID(ctypes.Structure):
    _fields_ = [('Data1',ctypes.c_ulong),('Data2',ctypes.c_ushort),
                ('Data3',ctypes.c_ushort),('Data4',ctypes.c_uint8*8)]
def _mk(s):
    s=s.strip('{}').upper().replace('-',''); g=_GUID()
    g.Data1=int(s[0:8],16); g.Data2=int(s[8:12],16); g.Data3=int(s[12:16],16)
    for i,b in enumerate(bytes.fromhex(s[16:])): g.Data4[i]=b; return g
def _hs(s):
    h=ctypes.c_void_p(); _combase.WindowsCreateString(s,len(s),ctypes.byref(h)); return h
def _qi(ptr, iid):
    if not ptr or not getattr(ptr,'value',None): return (-1, ctypes.c_void_p())
    out=ctypes.c_void_p()
    vt=ctypes.cast(ptr,ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn=ctypes.cast(ctypes.cast(vt,ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.WINFUNCTYPE(ctypes.c_long,ctypes.c_void_p,ctypes.POINTER(_GUID),ctypes.POINTER(ctypes.c_void_p)))
    hr=fn(ptr,ctypes.byref(iid),ctypes.byref(out)); return (hr, out)

IID_IAF = _mk('00000035-0000-0000-C000-000000000046')
IID_Ins = _mk('AF86E2E0-B12D-4C6A-9C5A-D7AA65101E90')

# Get GCI factory
h = _hs('Windows.Graphics.Capture.GraphicsCaptureItem')
fac = ctypes.c_void_p()
hr_f = _combase.RoGetActivationFactory(h, ctypes.byref(IID_IAF), ctypes.byref(fac))
_combase.WindowsDeleteString(h)
print(f'GCI factory: hr=0x{hr_f&0xFFFFFFFF:08X}  fac=0x{fac.value or 0:016X}')

# Try various IIDs from the pyd's list that might be IGraphicsCaptureItemStatics
candidate_iids = {
    'IGraphicsCaptureItemInterop': '3628E81B-3CAC-4C60-B7F4-23CE0E0C3356',
    'PYD_16760': '997439FE-F681-4A11-B416-C13A47E8BA36',
    'PYD_16770': '9C3654BC-4ADC-463A-B392-D6BC9289C925',
    'PYD_16790': '117E202D-A859-4C89-873B-C2AA566788E3',
    'PYD_167a0': '37869CFA-2B48-5EBF-9AFB-DFFD805DEFDB',
    'PYD_167b0': 'FA50C623-38DA-4B32-ACF3-FA9734AD800E',
    # These from the full pyd scan at the 0x168xx range:
    'PYD_16820': '5A1711B3-AD79-4B4A-9336-1318FDDE3539',
    'PYD_16830': '3B92ACC9-E584-5862-BF5C-9C316C6D2DBB',
    'PYD_16840': '2C39AE40-7D2E-5044-804E-8B6799D4CF9E',
    'PYD_16850': '814E42A9-F70F-4AD7-939B-FDDCC6EB880D',
    'PYD_16860': 'AE99813C-C257-5759-8ED0-668C9B557ED4',
    # Known Windows 10 IGraphicsCaptureItemStatics:
    'GCIStat v1': 'A87EBE20-AA92-4D27-B5C4-B9E3E3F98D8F',  # guessed
    # Windows 11 TryCreateFromWindowId interface (possible):
    'GCIStat v2': 'CBE4B04E-D93E-47AF-B9CC-5A7DC1E97EC1',  # guessed
}

print('\n--- QI factory for various potential statics IIDs ---')
for name, iid_str in candidate_iids.items():
    iid = _mk(iid_str)
    hr_q, out = _qi(fac, iid)
    if hr_q == 0 and out.value:
        print(f'  {name} ({iid_str}): FOUND → 0x{out.value:016X}')
        # Try calling vtable[6] with no output args to see what it does
        vt_o = ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value
        print(f'    vtable[6] fn = 0x{ctypes.cast(vt_o, ctypes.POINTER(ctypes.c_void_p))[6]:016X}')
        print(f'    vtable[7] fn = 0x{ctypes.cast(vt_o, ctypes.POINTER(ctypes.c_void_p))[7]:016X}')
    else:
        print(f'  {name}: NOT FOUND (0x{hr_q&0xFFFFFFFF:08X})')

# Try GetIids on the factory to see what statics it exposes
_CoTaskMemFree = ctypes.windll.ole32.CoTaskMemFree
_CoTaskMemFree.restype = None; _CoTaskMemFree.argtypes = [ctypes.c_void_p]
iid_count = ctypes.c_ulong(); iids_ptr = ctypes.c_void_p()
vt_fac = ctypes.cast(fac, ctypes.POINTER(ctypes.c_void_p)).contents.value
fn3 = ctypes.cast(ctypes.cast(vt_fac, ctypes.POINTER(ctypes.c_void_p))[3],
    ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_void_p)))
hr2 = fn3(fac, ctypes.byref(iid_count), ctypes.byref(iids_ptr))
print(f'\nFactory.GetIids: hr=0x{hr2&0xFFFFFFFF:08X}  count={iid_count.value}')
if hr2 == 0 and iids_ptr.value:
    raw = (ctypes.c_uint8 * (16 * iid_count.value)).from_address(iids_ptr.value)
    for i in range(iid_count.value):
        g = _GUID.from_buffer_copy(bytes(raw)[i*16:(i+1)*16])
        d4 = bytes(g.Data4)
        print(f'  [{i}] {g.Data1:08X}-{g.Data2:04X}-{g.Data3:04X}-{d4[:2].hex().upper()}-{d4[2:].hex().upper()}')
    _CoTaskMemFree(iids_ptr)
