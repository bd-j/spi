import numpy as np
from copy import deepcopy

bounds = {}
features = {}

def pad_bounds(inbounds, tpad=500., gpad=0.5, zpad=0.1, **extras):
    mm = np.array([-1,1])
    outbounds = {}
    outbounds['logt'] = tuple(np.log10(10**np.array(inbounds['logt']) + mm * tpad))
    outbounds['logg'] = tuple(np.array(inbounds['logg']) + mm*gpad)
    outbounds['feh'] = tuple(np.array(inbounds['feh']) + mm*gpad)
    return outbounds


#you get (order + 3 - 1)!/(order! * 2!) combinations for each order

# Warm Dwarfs
# 4000 < Teff <= 6500, logg > 3.5
bounds['Warm Dwarfs'] = {'logt': (np.log10(4000.0), np.log10(6300)),
                         'logg': (3.49, 5.5),
                         'feh': (-2.01, 0.51)}

features['Warm Dwarfs'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           # Quadratic
                           ['logt', 'logt'],
                           ['logg', 'logg'],
                           ['feh', 'feh'],
                           # Cross-quadratic
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['logg', 'feh'],
                           # Cubic
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh'],
                           # cross-cubic
                           ['logt', 'logt', 'logg'],
                           ['logt', 'logt', 'feh'],
                           ['logt', 'logg', 'logg'],
                           ['logt', 'feh', 'feh'],
                           ['logg', 'logg', 'feh'],
                           ['logg', 'feh', 'feh'],
                           ['logt', 'logg', 'feh'],
                           # Quartic
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh', 'feh'],
                           # cross-Quartic
                           ['logt', 'logt', 'logt', 'logg'],
                           ['logt', 'logt', 'logt', 'feh'],
                           ['logt', 'logt', 'logg', 'logg'],
                           ['logt', 'logt', 'feh', 'feh'],
                           ['logt', 'logg', 'logg', 'logg'],
                           ['logt', 'feh', 'feh', 'feh'],
                           ['feh', 'feh', 'logg', 'logg'],
                           ['feh', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh', 'logg'],
                           ['logt', 'logt', 'feh', 'logg'],
                           ['logt', 'feh', 'feh', 'logg'],
                           ['logt', 'feh', 'logg', 'logg'],
                           # Quintic
                           ['logt', 'logt', 'logt', 'logt', 'logt'],
                           #['logt', 'logt', 'logg', 'logg', 'logg'],
                           #['feh', 'feh', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'feh', 'feh'],
                           #['logt', 'logt', 'logt', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'feh', 'feh', 'feh'],
                           #['logt', 'logt', 'logt', 'logt', 'logt', 'logt']
                           )


# Warm Giants
# 4000 < Teff <~ 6500, logg <= 3.5
bounds['Warm Giants'] = {'logt': (np.log10(4000.0), np.log10(6300)),
                         'logg': (-0.25, 3.501),
                         'feh': (-2.01, 0.51)}
features['Warm Giants'] = ((['logt'],
                           ['feh'],
                           ['logg'],
                           # Quadratic
                           ['logt', 'logt'],
                           ['logg', 'logg'],
                           ['feh', 'feh'],
                           # Cross-quadratic
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['logg', 'feh'],
                           # Cubic
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh'],
                           # cross-cubic
                           ['logt', 'logt', 'logg'],
                           ['logt', 'logt', 'feh'],
                           ['logt', 'logg', 'logg'],
                           ['logt', 'feh', 'feh'],
                           ['logg', 'logg', 'feh'],
                           ['logg', 'feh', 'feh'],
                           ['logt', 'logg', 'feh'],
                           # Quartic
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh', 'feh'],
                           # cross-Quartic
                           ['logt', 'logt', 'logt', 'logg'],
                           ['logt', 'logt', 'logt', 'feh'],
                           ['logt', 'logt', 'logg', 'logg'],
                           ['logt', 'logt', 'feh', 'feh'],
                           ['logt', 'logg', 'logg', 'logg'],
                           ['logt', 'feh', 'feh', 'feh'],
                           ['feh', 'feh', 'logg', 'logg'],
                           ['feh', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh', 'logg'],
                           ['logt', 'logt', 'feh', 'logg'],
                           ['logt', 'feh', 'feh', 'logg'],
                           ['logt', 'feh', 'logg', 'logg'],
                           # Quintic
                           ['logt', 'logt', 'logt', 'logt', 'logt'],
                           #['logt', 'logt', 'logg', 'logg', 'logg'],
                           #['feh', 'feh', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'feh', 'feh'],
                           #['logt', 'logt', 'logt', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'feh', 'feh', 'feh'],
                           #['logt', 'logt', 'logt', 'logt', 'logt', 'logt']
                           )



bounds['Cool Giants'] = {'logt': (3.4, np.log10(4000.0)),
                         'logg': (-0.5, 2.25),
                         'feh': (-2.501, 0.5)}
# Teff <= 4000, logg <= 3.5 
features['Cool Giants'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           # quadratic
                           ['logt', 'logt'], 
                           ['logg', 'logg'],
                           ['feh', 'feh'],
                           # cross quadratic
                           ['feh', 'logg'],
                           ['logt', 'logg'],
                           ['logt', 'feh'],
                           # cubic
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           ['feh', 'feh', 'feh'],
                           # cross-cubic
                           ['logt', 'logg', 'feh'],
                           ['logt', 'logt', 'feh'],
                           ['logt', 'logt', 'logg'],
                           ['feh', 'feh', 'logt'],
                           ['feh', 'feh', 'logg'],
                           ['logt', 'logg', 'logg'],
                           ['feh', 'logg', 'logg'],
                           # quartic
                           ['logt', 'logt', 'logt', 'logt'],
                           #['logg', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logg', 'feh'],
                           #['logt', 'logg', 'logg', 'feh'],
                           #['feh', 'feh', 'feh', 'logg'],
                           #['feh', 'feh', 'logg', 'logg'],
                           # quintic
                           #['logt', 'logt', 'logt', 'logt', 'logt'],
                           #['logt', 'logt', 'logt', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'logt', 'logt', 'logt'],
                           #['logg', 'logg', 'logg', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'logt', 'logt', 'logt', 'logt', 'logt'],
                           #['logg', 'logg', 'logg', 'logg', 'logg', 'logg', 'logg'],
                           #['logt', 'logt', 'feh', 'feh', 'logg', 'logg', 'logg']
                           )

# Cool Dwarfs
# Teff <= 4000, logg > 3.5 
bounds['Cool Dwarfs'] = {'logt': (3.4, np.log10(4000.0)),
                         'logg': (3.999, 6),
                         'feh': (-2.501, 0.5)}
features['Cool Dwarfs'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           # Quadtratic
                           ['feh', 'feh'],
                           ['logt', 'logt'],
                           ['logg', 'logg'],
                           # Cross-quadratic
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['feh', 'logg'],
                           # Cubic
                           ['feh', 'feh', 'feh'],
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           # Cross-Cubic
                           ['logt', 'logt', 'feh'],
                           ['logt', 'feh', 'feh'],
                           ['logg', 'logt', 'logt'],
                           # Quartic
                           ['logt', 'logt', 'logt', 'logt'],
                           ['feh', 'feh', 'feh','feh'],
                           #['logg', 'logg', 'logg', 'logg'],
                           # Cross-Quartic
                           ['logt', 'logt', 'feh', 'feh'],
                           ['logt', 'logt', 'logt', 'feh'],
                           # Quintic
                           ['logt', 'logt', 'logt', 'logt', 'logt'],
                           )


#Hot Stars
bounds['Hot Stars'] = {'logt': (np.log10(6000), np.log10(12000.0)),
                         'logg': (2.999, 5.001),
                         'feh': (-2.501, 0.5)}
features['Hot Stars'] = (['logt'],
                          ['feh'],
                          ['logg'],
                          # Quadratic
                          ['logt', 'logt'],
                          ['feh', 'feh'],
                          ['logg', 'logg'],
                          # Cross-Quadratic
                          ['logt', 'logg'],
                          ['logt', 'feh'],
                          ['logg', 'feh'],
                          # cubic
                          ['logt', 'logt', 'logt'],
                          ['logg', 'logg', 'logg'],
                          ['feh', 'feh', 'feh'],
                          # cross-cubic
                          ['logt', 'logg', 'feh'],
                          ['logt', 'logt', 'feh'],
                          ['logt', 'logt', 'logg'],
                          ['feh', 'feh', 'logt'],
                          ['feh', 'feh', 'logg'],
                          ['logt', 'logg', 'logg'],
                          ['feh', 'logg', 'logg'],
                          # quartic
                          ['logt', 'logt', 'logt', 'logt'],
                          )

bounds['Very Hot Stars'] = {'logt': (np.log10(12000), np.log10(40000.0)),
                            'logg': (2.999, 5.001),
                            'feh': (-2.501, 0.5)}
