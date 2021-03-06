'''
Finding the spectral lines of Hydrogen for the Balmer - Alpha series.
'''
import numpy as np
import matplotlib.pyplot as plt

#parameters
m = 9.10938356 * 10**(-31) #kg
e0 = 8.85418782 * 10**(-12) #C^2/ kg m^3/s^2 = C^2 s^2/ kg m^3
hbar = 1.0545718 * 10**(-34)  #Js
e = 1.60217662 * 10**(-19) #C
h = hbar * 2 * np.pi * (6.24150913 * 10**(18)) #eV s
c = 2.997924580 * 10**8 * 10**9 # nm/s

#amount of energy levels
ni = np.linspace(1, 100, 100)
nf = np.linspace(2, 101, 100)

#Ground State Energy in eV
E = -(m/(2 * hbar**2)) * ((e**2)/(4 * np.pi * e0))**2 * (6.24150913 * 10**(18))
print(E)
#good, we get the correct G.S Energy for Hydrogen

#start list for Energy levels and wavelengths
En = []
wl = []
wlmaster = []

#nf>ni
#Compute the energy and wavelengths for each possible case.
for i in ni:
    for j in nf:
        if i == j:
            0
        if i > j:
            0
        if i < j:
             x = E * (1/(i**2) - 1/(j**2)) #transition energy for specific energy levels
             w = - (h*c)/x #transforms energy to wavelengths
             En.append(x) #fill the Energy list with possible energy levels
             wlmaster.append(w) # a master list of wavelengths for Hydrogen spectrum
             #pick only the values of visible light discard everything else from the master list
             if 390. <= w <= 770.:
                 wl.append(w) #list of the visible region
             else:
                 0

wl = np.asarray(wl)
En = np.asarray(En)
wlmaster = np.asarray(wlmaster)

print('Minimum wavelength is',np.min(wl),'and Maximum wavelength is', np.max(wl))


plt.figure(); plt.hist(wlmaster,bins = 200,range = (0,10**3),label = 'Wavelength');plt.xlabel('Wavelength (nm)'); plt.show() #a hist generated from the master list
plt.figure(); plt.hist(wl,bins = 200,range = (380,700),label = 'Wavelength'); plt.xlabel('Wavelength (nm)'); plt.show() # a hist generated from the visible light list
