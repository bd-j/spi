## I’ve got a long list of MILES stars that are “bad” in one way or another.  e.g.,

## HD206778
## HD225212
## HD163990
## HD080390

## and:

##   ;remove these stars b/c Teff seems to be wrong
##   spec[819-1,*] = 0.0 ;HD207673, A2Ib
##   spec[140-1,*] = 0.0 ;HD281679, B8 D; PNe!
##   spec[834-1,*] = 0.0 ;HD213307, A0 D (Star suspected of Variability)
##   spec[104-1,*] = 0.0 ;HD018391, G0Ia
##   spec[593-1,*] = 0.0 ;HD147923, S star
##   spec[386-1,*] = 0.0 ;HD089822B, A0sp?
##   spec[933-1,*] = 0.0 ;M4_LEE-2303
##   spec[234-1,*] = 0.0 ;HD049331, M1Iab variable
##   spec[617-1,*] = 0.0 ;HD156283 K3Iab variable
##   spec[99-1,*]  = 0.0 ;HD017491, M5III pulsating star

## these should definitely be removed from the interpolator!

## you might also want to remove these:

##   ;remove binaries/multiples
##   spec[256-1,*] = 0.0 ;HD059612
##   spec[78-1,*]  = 0.0 ;HD013267
##   spec[252-1,*] = 0.0 ;HD057061
##   spec[572-1,*] = 0.0 ;HD141851
##   spec[836-1,*] = 0.0 ;HD213470, A3Ia
##   spec[678-1,*] = 0.0 ;HD169985, A0Vs+G:III

milesid_bad_teff = [819, 140, 834, 104, 593, 386, 933, 234, 617, 99]
milesid_multiples = [256, 78, 252, 572, 836, 678]
milesid_badid_sharma = [501, 952, 591]
milesid_bad_sharma = {175: 'variable' 459: 'M8III pulsating', 508: 'M8 pulsating',
                      580: 'BYDraconis variable, P11 params ok',
                      593: 'mis-id?', 838: 'ok', 884: 'flaring M dwarf with emission lines',
                      890: 'ok', 927: 'mis-id? (M4 cluster)', 964: 'contaminated',
                      972: 'P11 params ok', 980: 'P11 params ok', 220: 'P11 params ok',
                      250: 'P11 params ok', 400: 'metallicity off?', 784: 'P11 params ok',
                      941: 'P11 params ok', 934: 'cluster interloper, wrong feh',
                      967: 'possibly swapped name with 968, not a cluster member',
                      968: 'P11 params ok',
                      }

allbad = milesid_bad_teff + milesid_multiple + milesid_badid_sharma + milesid_bad_sharma.keys()


