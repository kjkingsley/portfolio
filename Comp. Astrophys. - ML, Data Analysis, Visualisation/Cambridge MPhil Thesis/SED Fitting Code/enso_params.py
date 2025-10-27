import sys
import os
import numpy as np
from astropy.io import fits
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.models.sedmodel import SpecModel
from prospect.observation import Photometry, Lines, Spectrum
from prospect.io import write_results as writer
from astropy import cosmology
cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
from astropy.table import Table
import sedpy
import pandas as pd

from matplotlib.pyplot import *

# %matplotlib inline

# re-defining plotting defaults
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 30})

# paths
# path_wdir = "/home/kkingsley/enso"
path_wdir = "/home/kkingsley/enso"
# path_cats = os.path.join(path_wdir, "catalogs")
path_output = os.path.join(path_wdir, "runs/")


# define dicts - DATA

def build_obs(component, err_floor=0.05, err_floor_el=0.05, **kwargs):
    """Build a dictionary of observational data. 
    
    :param snr:
        The S/N to assign to the photometry, since none are reported 
        in Johnson et al. 2013
        
    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes 
        into apparent magnitudes.
        
    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    # from prospect.utils.obsutils import fix_obs # OUTDATED


    obs = {}

    filter_list = ["F090W", "F115W", "F150W", "F182M", "F200W", "F210M", "F277W", "F356W", "F410M", "F444W"]
    obs_filters = sedpy.observate.load_filters(["jwst_" + f.lower() for f in filter_list])
    obs["filters"] = obs_filters


    z_spec = 5.3861

    # Now we store the measured fluxes for each filter, and their uncertainties.
### Server side fluxes ###
    df = pd.read_csv("/home/kkingsley/enso/4compSVI_fluxes_nJy.csv")
### Local fluxes ###
    # df = pd.read_csv("/home/kings/prospector/Enso/data/4compSVI_fluxes_nJy.csv")


## Component 0 Fluxes: ##
    # load in fluxes as nJy, convert to Jy then magggies
    flux_index = 'flux_{}s'.format(component)
    lower_index = 'flux_{}s_delta_lower'.format(component)
    upper_index = 'flux_{}s_delta_upper'.format(component)

    maggies = df[flux_index].to_numpy() * 1e-9 / 3631
    obs["maggies"] = maggies

    # creating uncertainties from upper and lower errors
    f0u_l = df[lower_index].to_numpy() * 1e-9 / 3631
    f0u_u = (df[upper_index]).to_numpy() * 1e-9 / 3631
    lbound = maggies - f0u_l
    ubound = maggies + f0u_u

    flux_unc = (ubound - lbound) / 2

    # And now we store the uncertainties (again in units of maggies)
    maggies_unc = np.array(flux_unc)
    obs["maggies_unc"] = maggies_unc

    print("###############################\n")
    print("Component: {}.\nMaggies: {}.\nUncertainties: {}.".format(component, maggies, maggies_unc))
    print("###############################")


## Build rest of obs dict ##
    # Now we need a mask, which says which flux values to consider in the likelihood.
    phot_mask = np.array([obs["filters"]])
    # obs["phot_mask"] = phot_mask
    obs["phot_mask"] = None

    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = None
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = None
    # (spectral uncertainties are given here)
    obs['unc'] = None
    # (again, to ignore a particular wavelength set the value of the 
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = None

    obs_wave_eff = [f.wave_effective for f in obs_filters]

    pdat = Photometry(
            filters =   obs_filters,
            flux =   maggies,
            uncertainty = maggies_unc,
            mask    =   np.isfinite(maggies) #phot_mask & (np.array(obs_wave_eff) > ((z_spec + 1) * 912))
                    )
    
    # sdat = Spectrum()
    # sdat.rectify()
    pdat.rectify()

    return [pdat] #, sdat]

# --------------
# Model Definition 
# --------------

def build_model(object_redshift=5.3861, sfh_template="continuity_sfh", nbins_sfh=8, student_t_width=0.3, z_limit_sfh=20.0, 
                add_IGM_model=False, add_duste=False, add_agn=False, add_neb=False, **extras):
    """
    Construct a model.
    sfh_template : "continuity_sfh", "dirichlet_sfh", "parametric_sfh"

    """
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
    from prospect.models import priors, sedmodel
    from prospect.models import transforms
    # from prospect.models.sedmodel import SedModel


    # get SFH template
    if (sfh_template == "continuity_sfh"):
        model_params = TemplateLibrary["continuity_sfh"]
    elif (sfh_template == "dirichlet_sfh"):
        model_params = TemplateLibrary["dirichlet_sfh"]
    elif (sfh_template == "parametric_sfh"):
        model_params = TemplateLibrary["parametric_sfh"]

    # IMF: 0: Salpeter (1955); 1: Chabrier (2003); 2: Kroupa (2001)
    model_params['imf_type']['init'] = 1

    # fit for redshift
    model_params["zred"]['isfree'] = False
    
    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    # add redshift scaling to agebins, such that
    def zred_to_agebins(zred=model_params["zred"]["init"], agebins=None, z_limit_sfh=20.0, nbins_sfh=8, **extras):
        tuniv = cosmo.age(zred).value*1e9
        tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9
        # tbinmax = tuniv*0.95
        agelims = np.append(np.array([0.0, 6.7, 7.0, 7.4772]), np.linspace(7.4772, np.log10(tbinmax), int(nbins_sfh-2))[1:])
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=model_params["zred"]["init"], nbins_sfh=8, z_limit_sfh=None, **extras):
        agebins = zred_to_agebins(zred=zred, nbins_sfh=nbins_sfh)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        nbins = agebins.shape[0]
        sratios = 10**logsfr_ratios
        dt = (10**agebins[:, 1]-10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

    if (sfh_template == "continuity_sfh"):
        # adjust number of bins for SFH and prior
        model_params['agebins']['N'] = nbins_sfh
        model_params['mass']['N'] = nbins_sfh
        model_params['logsfr_ratios']['N'] = nbins_sfh-1
        model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1, 0.0)  # constant SFH
        model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0), scale=np.full(nbins_sfh-1, student_t_width), df=np.full(nbins_sfh-1, 2))
        model_params['agebins']['depends_on'] = zred_to_agebins
        # set mass prior
        model_params["logmass"]["prior"] = priors.TopHat(mini=6, maxi=12)
        model_params['mass']['depends_on'] = logmass_to_masses

    elif (sfh_template == "dirichlet_sfh"):
        tuniv = cosmo.age(model_params["zred"]["init"]).value*1e9
        tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9
        agelims = np.append(np.array([0.0, 6.7, 7.0, 7.4772]), np.linspace(7.4772, np.log10(tbinmax), int(nbins_sfh-2))[1:])
        model_params = adjust_dirichlet_agebins(model_params, agelims=agelims)
        model_params["total_mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)

    elif (sfh_template == "parametric_sfh"):
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
        model_params["tage"]["prior"] = priors.TopHat(mini=1e-3, maxi=cosmo.age(model_params["zred"]["init"]).value)
        model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)

    # adjust other priors
    model_params["logzsol"]["prior"] = priors.ClippedNormal(mean=-1.0, sigma=0.3, mini=-2.0, maxi=0.19)  # priors.TopHat(mini=-2.0, maxi=0.19)

    # complexify the dust
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=6.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1,
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return(dust1_fraction*dust2)

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # fit for IGM absorption
    if add_IGM_model:
        model_params.update(TemplateLibrary["igm"])
        model_params["igm_factor"]['isfree'] = True
        model_params["igm_factor"]["prior"] = priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)
    else:
        model_params.update(TemplateLibrary["igm"])
        model_params["igm_factor"]['isfree'] = False

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_umin']['isfree'] = True

    if add_agn:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['agn_tau']['isfree'] = True

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        #model_params["gas_logz"]["depends_on"] = transforms.stellar_logzsol

    # if add_eline_scaling:
    #     # Rescaling of emission lines
    #     model_params["linespec_scaling"] = {"N": 1,
    #                                         "isfree": True,
    #                                         "init": 1.0, "units": "multiplative rescaling factor",
    #                                         "prior": priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.1, maxi=2.0)}

    # Now instantiate the model using this new dictionary of parameter specifications

    model = sedmodel.SpecModel(model_params) #LineSpecModel doesn't exist anymore
    # model.params  # will reflect dependencies after this
    # model.set_parameters(model.rectify_theta(model.theta))
    return model

run_params = {}
run_params["object_redshift"] = 5.3861

model = build_model(**run_params)
print(model)
print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
print("Initial parameter dictionary:\n{}".format(model.params))


# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, sfh_template="continuity_sfh", compute_vega_mags=False, **extras):
    if (sfh_template == "continuity_sfh") or (sfh_template == "dirichlet_sfh"):
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=zcontinuous,
                            compute_vega_mags=compute_vega_mags,
                            reserved_params=['tage', 'sigma_smooth'])
    else:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags,
                           reserved_params=['sigma_smooth'])
    return sps


# -----------------
# Noise Model
# ------------------

def build_noise(observations, **extras):
    return observations


# -----------
# Everything
# ------------

def build_all(**kwargs):
    observations = build_obs(**kwargs)
    observations = build_noise(observations, **kwargs)
    model = build_model(**kwargs)
    sps = build_sps(**kwargs)

    return (observations, model, sps)


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--component', type=int, default=0,
                        help="Specified component to fit.")
    parser.add_argument('--incl_hst', action="store_true",
                        help="If set, include HST photometry.")
    parser.add_argument('--fit_el', action="store_true",
                        help="If set, fit emission lines.")
    parser.add_argument('--sfh_template', type=str, default="continuity_sfh",
                        help="SFH template assumed: continuity_sfh, dirichlet_sfh or parametric_sfh.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_agn', action="store_true",
                        help="If set, add AGN emission to the model.")
    parser.add_argument('--add_IGM_model', action="store_true",
                        help="If set, add flexibility to IGM model (scaling of Madau attenuation).")
    parser.add_argument('--add_eline_scaling', action="store_true",
                        help="If set, add flexibility to IGM model (scaling of Madau attenuation).")
    parser.add_argument('--objid', type=int, default=1,
                        help="1-index row number in the table to fit.")
    parser.add_argument('--nbins_sfh', type=int, default=8,
                        help="Number of SFH bins.")
    parser.add_argument('--z_limit_sfh', type=float, default=20,
                        help="Redshift when SFH starts.")
    parser.add_argument('--student_t_width', type=float, default=0.3,
                        help="Width of student-t distribution.")
    parser.add_argument('--err_floor', type=float, default=0.05,
                        help="Error floor for photometry.")
    parser.add_argument('--err_floor_el', type=float, default=0.05,
                        help="Error floor for EL.")

    
    
    args = parser.parse_args()


   # --- Configure ---
    run_params = vars(args)
    

    # add in dynesty settings
    run_params['nested_sampler']= 'dynesty'
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 2000
    run_params['nested_dlogz_init'] = 0.02
    run_params['nested_maxcall'] = 2500000
    run_params['nested_maxcall_init'] = 2500000
    run_params['nested_sample'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_save_bounds'] = False
    # ! Shrunk for testing:
    run_params['nested_target_n_effective'] = 10000 #200 000 for amanda
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    # run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}

    print(run_params)
    
    run_params["param_file"] = __file__

     # --- Get fitting ingredients ---
    obs, model, sps = build_all(**run_params)
    print("#########################")
    print(obs)
    print("#########################")
    print(model)
    print("#########################")
    print(model.predict(model.theta, observations=obs, sps=sps))
    print("#########################")

    if args.debug:
        sys.exit()

    # --- Fit the model ---
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = path_output + "{2}_{0}_{1}_mcmc.h5".format(args.outfile, "enso", args.component)
    output = fit_model(obs, model, sps, **run_params)

    print("writing to {}".format(hfile))

    writer.write_hdf5(hfile,
                      run_params,
                      model,
                      obs,
                      output["sampling"],
                      output["optimization"],
                      sps=sps
                      )

    try:
        hfile.close()
    except(AttributeError):
        pass
