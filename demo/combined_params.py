import numpy as np

bounds = {}
features = {}

bounds['Cool Giants'] = {'logt': (3.4, np.log10(4000.0)),
                         'logg': (-0.5, 2.25),
                         'feh': (-3, 0.5)}
# Teff <= 4000, logg <= 3.5 
features['Cool Giants'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           ['logt', 'logt'], 
                           ['logt', 'logg'],
                           ['logt', 'feh'], 
                           ['logg', 'logg'],
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           ['logt', 'logg', 'feh'],
                           ['logt', 'feh', 'feh'],
                           ['feh', 'feh', 'feh'],
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logg', 'feh'],
                           ['logt', 'logg', 'logg', 'feh'],
                           ['feh', 'feh', 'feh', 'logg'],
                           ['feh', 'feh', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt', 'logt'],
                           ['logt', 'logt', 'logt', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt', 'logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg', 'logg', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'feh', 'feh', 'logg', 'logg', 'logg']
                           )

# Cool Dwarfs
# Teff <= 4000, logg > 3.5 
bounds['Cool Dwarfs'] = {'logt': (3.4, np.log10(4000.0)),
                         'logg': (4.001, 6),
                         'feh': (-3, 0.5)}
features['Cool Dwarfs'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           ['logt', 'logt'],
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['feh', 'feh'],
                           ['logt', 'logt', 'feh'],
                           ['logt', 'feh', 'feh'],
                           ['feh', 'feh', 'feh'],
                           ['logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt'],
                           ['logt', 'logt', 'feh', 'feh'],
                           ['feh', 'feh', 'feh','feh'],
                           ['logg', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logt', 'logt', 'logt', 'logt', 'logt']
                           )

# Warm Giants
# 4000 < Teff <= 6500, logg <= 3.5
# should also remove very warm low-gravity C3K 
bounds['Warm Giants'] = {'logt': (np.log10(4000.0), np.log10(6000)),
                         'logg': (-0.25, 3.501),
                         'feh': (-3, 0.5)}
features['Warm Giants'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           ['logt', 'logt'],
                           ['logg', 'logg'],
                           ['feh', 'feh'],
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['logt', 'logt', 'logt'],
                           ['logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logt', 'logt', 'feh', 'feh'],
                           ['logt', 'logt', 'logg', 'logg'],
                           ['feh', 'feh', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt', 'logt']
                           )

# Warm Dwarfs
# 4000 < Teff <= 6500, logg > 3.5
bounds['Warm Dwarfs'] = {'logt': (np.log10(4000.0), np.log10(6000)),
                         'logg': (3.500, 5.25),
                         'feh': (-3, 0.5)}
features['Warm Dwarfs'] = (['logt'],
                           ['feh'],
                           ['logg'],
                           ['logt', 'logt'],
                           ['logg', 'logg'],
                           ['feh', 'feh'],
                           ['logt', 'feh'],
                           ['logt', 'logg'],
                           ['logt', 'logt', 'logt'],
                           ['logt', 'logt', 'logg'],
                           ['logt', 'logt', 'feh'],
                           ['logt', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt'],
                           ['logt', 'logg', 'logg', 'logg'],
                           ['feh', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'logg', 'logg'],
                           ['feh', 'feh', 'logt', 'logg'],
                           ['feh', 'feh', 'logt', 'logt'],
                           ['logt', 'logt', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'logt', 'logt'],
                           ['logt', 'logt', 'logg', 'logg', 'logg'],
                           ['feh', 'feh', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'feh', 'feh'],
                           ['logt', 'logt', 'logt', 'logg', 'logg', 'logg'],
                           ['logt', 'logt', 'logt', 'feh', 'feh', 'feh'],
                           ['logt', 'logt', 'logt', 'logt', 'logt', 'logt']
                           )

# Hot Giants
# Teff > 6500, logg <= 3.5
# only 3 non-c3k stars
bounds['Hot Giants'] = {'logt': (np.log10(6000), 4.2),
                         'logg': (2.0, 3.500),
                         'feh': (-3, 0.5)}
features['Hot Giants'] = (['logt'],
                          ['feh'],
                          ['logg'],
                          ['logt', 'logt'],
                          ['feh', 'feh'],
                          ['logg', 'logg']
                          )

# Hot Dwarfs
# Teff > 6500, logg > 3.5 
bounds['Hot Dwarfs'] = {'logt': (np.log10(5800), 4.2),
                         'logg': (3.500, 5.25),
                         'feh': (-3, 0.5)}
features['Hot Dwarfs'] = (['logt'],
                          ['feh'],
                          ['logg'],
                          ['logt', 'logt'],
                          ['feh', 'feh'],
                          ['logg', 'logg']
                          )
