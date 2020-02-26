from sandbox import *


quarks = [
    "up",
    "down",
    "charm",
    "strange",
    "top",
    "bottom",
    "antiup",
    "antidown",
    "anticharm",
    "antistrange",
    "antitop",
    "antibottom",
]
leptons = [
    "electron",
    "electron neutrino",
    "muon",
    "muon neutrino",
    "tau",
    "tau neutrino",
]
bosons = ["photon", "w boson", "z boson", "gluon", "higgs boson"]

mass = {
    "up": 2.3 * ur.MeV * ur.c ** (-2),
    "down": 4.8 * ur.MeV * ur.c ** (-2),
    "charm": 1275 * ur.MeV * ur.c ** (-2),
    "strange": 95 * ur.MeV * ur.c ** (-2),
    "top": 173210 * ur.MeV * ur.c ** (-2),
    "bottom": 4180 * ur.MeV * ur.c ** (-2),
    "antiup": 2.3 * ur.MeV * ur.c ** (-2),
    "antidown": 4.8 * ur.MeV * ur.c ** (-2),
    "anticharm": 1275 * ur.MeV * ur.c ** (-2),
    "antistrange": 95 * ur.MeV * ur.c ** (-2),
    "antitop": 173210 * ur.MeV * ur.c ** (-2),
    "antibottom": 4180 * ur.MeV * ur.c ** (-2),
    "electron": 0.5109989461 * ur.MeV * ur.c ** (-2),
    "electron neutrino": 0.0000022 * ur.MeV * ur.c ** (-2),
    "muon": 105.6583745 * ur.MeV * ur.c ** (-2),
    "muon neutrino": 0.170 * ur.MeV * ur.c ** (-2),
    "tau": 1776.86 * ur.MeV * ur.c ** (-2),
    "tau neutrino": 15.5 * ur.MeV * ur.c ** (-2),
    "photon": 0 * ur.GeV * ur.c ** (-2),
    "w boson": 80.385 * ur.GeV * ur.c ** (-2),
    "z boson": 91.1875 * ur.GeV * ur.c ** (-2),
    "gluon": 0 * ur.GeV * ur.c ** (-2),
    "higgs boson": 125.09 * ur.GeV * ur.c ** (-2),
    "graviton": 0 * ur.GeV * ur.c ** (-2),
}
total_angular_momentum = {
    "up": 1 / 2,
    "down": 1 / 2,
    "charm": 1 / 2,
    "strange": 1 / 2,
    "top": 1 / 2,
    "bottom": 1 / 2,
    "antiup": 1 / 2,
    "antidown": 1 / 2,
    "anticharm": 1 / 2,
    "antistrange": 1 / 2,
    "antitop": 1 / 2,
    "antibottom": 1 / 2,
}
baryon_number = {
    "up": 1 / 3,
    "down": 1 / 3,
    "charm": 1 / 3,
    "strange": 1 / 3,
    "top": 1 / 3,
    "bottom": 1 / 3,
    "antiup": -1 / 3,
    "antidown": -1 / 3,
    "anticharm": -1 / 3,
    "antistrange": -1 / 3,
    "antitop": -1 / 3,
    "antibottom": -1 / 3,
}
electric_charge = {
    "up": 2 / 3 * ur.e,
    "down": -1 / 3 * ur.e,
    "charm": 2 / 3 * ur.e,
    "strange": -1 / 3 * ur.e,
    "top": 2 / 3 * ur.e,
    "bottom": -1 / 3 * ur.e,
    "antiup": -2 / 3 * ur.e,
    "antidown": 1 / 3 * ur.e,
    "anticharm": -2 / 3 * ur.e,
    "antistrange": 1 / 3 * ur.e,
    "antitop": -2 / 3 * ur.e,
    "antibottom": 1 / 3 * ur.e,
    "electron": -1 * ur.e,
    "electron neutrino": 0 * ur.e,
    "muon": -1 * ur.e,
    "muon neutrino": 0 * ur.e,
    "tau": -1 * ur.e,
    "tau neutrino": 0 * ur.e,
    "photon": 0 * ur.e,
    "w boson": -1 * ur.e,
    "z boson": 0 * ur.e,
    "gluon": 0 * ur.e,
    "higgs boson": 0 * ur.e,
    "graviton": 0 * ur.e,
}
spin = {
    "photon": 1,
    "w boson": 1,
    "z boson": 1,
    "gluon": 1,
    "higgs boson": 0,
    "graviton": 2,
}
weak_isospin_3 = {
    "up": 1 / 2,
    "down": -1 / 2,
    "charm": 0,
    "strange": 0,
    "top": 0,
    "bottom": 0,
    "antiup": -1 / 2,
    "antidown": 1 / 2,
    "anticharm": 0,
    "antistrange": 0,
    "antitop": 0,
    "antibottom": 0,
}
charm = {
    "up": 0,
    "down": 0,
    "charm": 1,
    "strange": 0,
    "top": 0,
    "bottom": 0,
    "antiup": 0,
    "antidown": 0,
    "anticharm": -1,
    "antistrange": 0,
    "antitop": 0,
    "antibottom": 0,
}
strangeness = {
    "up": 0,
    "down": 0,
    "charm": 0,
    "strange": -1,
    "top": 0,
    "bottom": 0,
    "antiup": 0,
    "antidown": 0,
    "anticharm": 0,
    "antistrange": 1,
    "antitop": 0,
    "antibottom": 0,
}
topness = {
    "up": 0,
    "down": 0,
    "charm": 0,
    "strange": 0,
    "top": 1,
    "bottom": 0,
    "antiup": 0,
    "antidown": 0,
    "anticharm": 0,
    "antistrange": 0,
    "antitop": -1,
    "antibottom": 0,
}
bottomness = {
    "up": 0,
    "down": 0,
    "charm": 0,
    "strange": 0,
    "top": 0,
    "bottom": -1,
    "antiup": 0,
    "antidown": 0,
    "anticharm": 0,
    "antistrange": 0,
    "antitop": 0,
    "antibottom": 1,
}

charges = [
    mass,
    electric_charge,
]
