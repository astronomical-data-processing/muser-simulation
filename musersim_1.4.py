# %matplotlib inline

import os
import sys
import numpy
import matplotlib
matplotlib.use('Agg')

from matplotlib import pylab

from matplotlib import pyplot as plt
# from matplotlib import plt.savefig
from astropy.coordinates import EarthLocation, SkyCoord

pylab.rcParams['figure.figsize'] = (10.0, 10.0)
pylab.rcParams['image.cmap'] = 'rainbow'

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

# sys.path.append(os.path.join('..','..'))

from data_models.parameters import arl_path
# results_dir = arl_path('test_results')

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord

from data_models.polarisation import PolarisationFrame

from wrappers.serial.image.iterators import image_raster_iter

from wrappers.serial.visibility.base import create_visibility
from wrappers.serial.visibility.operations import sum_visibility
from wrappers.serial.visibility.iterators import vis_timeslices, vis_wslices
from wrappers.serial.simulation.configurations import create_configuration_from_file
from wrappers.serial.skycomponent.operations import create_skycomponent, find_skycomponents, \
    find_nearest_skycomponent, insert_skycomponent
from wrappers.serial.image.operations import show_image, export_image_to_fits, qa_image, smooth_image
from wrappers.serial.imaging.base import advise_wide_field, create_image_from_visibility, \
    predict_skycomponent_visibility

from wrappers.arlexecute.griddata.kernels import create_awterm_convolutionfunction
from wrappers.arlexecute.griddata.convolution_functions import apply_bounding_box_convolutionfunction
from wrappers.serial.griddata.kernels import create_awterm_convolutionfunction
from wrappers.serial.griddata.convolution_functions import apply_bounding_box_convolutionfunction

# Use workflows for imaging
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from workflows.shared.imaging.imaging_shared import imaging_contexts
from workflows.arlexecute.imaging.imaging_arlexecute import predict_list_arlexecute_workflow, \
    invert_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import predict_list_serial_workflow, \
    invert_list_serial_workflow

import logging
from data_models.parameters import arl_path

# results_dir = arl_path('test_results')
dask_dir = BASE_DIR+"/dask-work-space/" #arl_path('test_results/dask-work-space')
if not os.path.isdir(dask_dir):
    os.mkdir(dask_dir)

def init_logging():
    logging.basicConfig(filename='%s/muser-pipeline.log' % 'result',
                        filemode='a',
                        format='%%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

def create_configuration(name: str = 'LOWBD2', **kwargs):
    location = EarthLocation(lon=115.2505, lat=42.211833333, height=1365.0)
    lowcore = create_configuration_from_file(antfile="muser-1.csv",
                                            mount='altaz', names='MUSER_%d',
                                            diameter=4.0, name='MUSER', location=location, **kwargs)
    return lowcore

if __name__ == '__main__':

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))

    pylab.rcParams['figure.figsize'] = (12.0, 12.0)
    pylab.rcParams['image.cmap'] = 'rainbow'
    pylab.rcParams['font.size'] = 12

    lowcore = create_configuration('MUSER')
    # lowcore = create_named_configuration('MUSER')
    # arlexecute.set_client(use_dask=True)
    # arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
    #                       local_dir=dask_dir)
    times = numpy.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * (numpy.pi / 12.0)
    frequency = numpy.array([14e8])
    storedir = str(frequency//1e6)
    channel_bandwidth = numpy.array([25e6])
    reffrequency = numpy.max(frequency)
    phasecentre = SkyCoord(ra=+80 * u.deg, dec=15 * u.deg, frame='icrs', equinox='J2000')
    vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                        weight=1.0, phasecentre=phasecentre,
                        polarisation_frame=PolarisationFrame('stokesI'))

    # print("##################", vt.data)
    advice = advise_wide_field(vt, wprojection_planes=4)

    # plt.clf()
    plt.plot(vt.data['uvw'][:, 0], vt.data['uvw'][:, 1], '.', color='b')
    plt.plot(-vt.data['uvw'][:, 0], -vt.data['uvw'][:, 1], '.', color='r')
    plt.xlabel('U (wavelengths)')
    plt.ylabel('V (wavelengths)')
    plt.title("UV coverage")
    plt.savefig('result/UV_coverage.jpg')
    # plt.show()

    # vt.data['vis'] *= 0.0
    npixel=1024

    # print("*****************", vt.data['vis'])

    model = create_image_from_visibility(vt, npixel=npixel,  nchan=1,
                                        polarisation_frame=PolarisationFrame('stokesI')) #cellsize=2*3.41e-5,

    # show_image(model)
    # plt.title("Original dirty image")
    # plt.savefig('result/400_0.jpg')

    centre = model.wcs.wcs.crpix-1
    spacing_pixels = npixel // 8
    # For an observation for the Sun, the disk is about 34 arcmin, if total FOV is 60 arcmin, then
    #   the disk edge would be 34/60*8
    log.info('Spacing in pixels = %s' % spacing_pixels)
    spacing = model.wcs.wcs.cdelt * spacing_pixels
    # locations = [-3.5, -2.5, -2.27, -1.5, -1., -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
    locations = [-3.5, -3.0, -2.3, -1.5,-0.5, 0.5, 1.5, 2.3, 3.0, 3.5]

    original_comps = []
    # We calculate the source positions in pixels and then calculate the
    # world coordinates to put in the skycomponent description
    for iy in locations:
        for ix in locations:
            if ix >= iy: #True:
                p = int(round(centre[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                    int(round(centre[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                sc = pixel_to_skycoord(p[0], p[1], model.wcs)
                # log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
                flux = numpy.array([[100.0 + 2.0 * ix + iy * 20.0]])  #numpy.array([[200.0]])
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                        polarisation_frame=PolarisationFrame('stokesI'))
                original_comps.append(comp)
                insert_skycomponent(model, comp)


    predict_skycomponent_visibility(vt, original_comps)

    print("*****************", vt.data['vis'])

    show_image(model)
    plt.title("Original dirty image")
    plt.savefig('result/400_0.jpg')

    cmodel = smooth_image(model)
    show_image(cmodel)
    plt.title("Smoothed model image")
    plt.savefig('result/400_1.jpg')
    #plt.show()

    comps = find_skycomponents(model, fwhm=1.0, threshold=10.0, npixels=5)
    plt.clf()
    for i in range(len(comps)):
        ocomp, sep = find_nearest_skycomponent(comps[i].direction, original_comps)
        # if i%10==2 or i%10==7:
        #     print("**************** Distortion *******************:")
        print(comps[i].direction.ra.value  - ocomp.direction.ra.value, comps[i].direction.dec.value - ocomp.direction.dec.value, comps[i].flux, ocomp.flux, '\n')
        plt.plot((comps[i].direction.ra.value  - ocomp.direction.ra.value)/cmodel.wcs.wcs.cdelt[0],
                (comps[i].direction.dec.value - ocomp.direction.dec.value)/cmodel.wcs.wcs.cdelt[1],
                '.', color='r')

    plt.xlabel('delta RA (pixels)')
    plt.ylabel('delta DEC (pixels)')
    plt.title("Recovered - Original position offsets")
    plt.savefig('result/400_2.jpg')
    # plt.show()

    wstep = 8.0
    # nw = int(1.1 * 800/wstep)
    nw1 = int(1.1 * 2.0 * numpy.max(numpy.abs(vt.w)) / wstep)
    # gcfcf = create_awterm_convolutionfunction(model, nw=110, wstep=8, oversampling=8,

    gcfcf = create_awterm_convolutionfunction(model,nw=nw1,wstep=8,oversampling=8,support=6,use_aaf=True)

    cf=gcfcf[1]
    plt.clf()
    plt.imshow(numpy.real(cf.data[0,0,0,0,0,:,:]))
    plt.title(str(numpy.max(numpy.abs(cf.data[0,0,0,0,0,:,:]))))
    plt.savefig('result/400_3.jpg')
    # plt.show()

    cf_clipped = apply_bounding_box_convolutionfunction(cf, fractional_level=1e-3)
    # print(cf_clipped.data.shape)
    gcfcf_clipped=(gcfcf[0], cf_clipped)

    plt.clf()
    plt.imshow(numpy.real(cf_clipped.data[0,0,0,0,0,:,:]))
    plt.title(str(numpy.max(numpy.abs(cf_clipped.data[0,0,0,0,0,:,:]))))
    plt.savefig('result/400_4.jpg')
    # plt.show()

    contexts = imaging_contexts().keys()
    print(contexts)

    #dict_keys(['facets_wstack', 'facets_timeslice', '2d', 'wstack', 'wprojection', 'timeslice', 'facets', 'wsnapshots'])

    print(gcfcf_clipped[1])

    # contexts = ['2d', 'facets', 'timeslice', 'wstack', 'wprojection']
    contexts = ['2d', 'facets', 'wprojection']

    for context in contexts:

        print('Processing context %s' % context)

        vtpredict_list =[create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
            weight=1.0, phasecentre=phasecentre, polarisation_frame=PolarisationFrame('stokesI'))]
        model_list = [model]
        # vtpredict_list = arlexecute.compute(vtpredict_list, sync=True)
        # vtpredict_list = arlexecute.scatter(vtpredict_list)

        if context == 'wprojection':
            future = predict_list_serial_workflow(vtpredict_list, model_list, context=context, facets=4)  #gcfcf=[gcfcf_clipped]

        elif context == 'facets':
            future = predict_list_serial_workflow(vtpredict_list, model_list, context=context, facets=4)

        # elif context == 'timeslice':
        #     future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context, vis_slices=vis_timeslices(
        #         vtpredict, 'auto'))
        #
        # elif context == 'wstack':
        #     future = predict_list_arlexecute_workflow(vtpredict_list, model_list, context=context, vis_slices=31)

        else:
            future = predict_list_serial_workflow(vtpredict_list, model_list, context=context)

        # vtpredict_list = arlexecute.compute(future, sync=True)

        vtpredict = vtpredict_list[0]

        uvdist = numpy.sqrt(vt.data['uvw'][:, 0] ** 2 + vt.data['uvw'][:, 1] ** 2)
        plt.clf()
        # print("#######vis############:", numpy.abs(vt.data['vis'][:]))
        plt.plot(uvdist, numpy.abs(vt.data['vis'][:]), '.', color='r', label="DFT")

        plt.plot(uvdist, numpy.abs(vtpredict.data['vis'][:]), '.', color='b', label=context)
        plt.plot(uvdist, numpy.abs(vtpredict.data['vis'][:] - vt.data['vis'][:]), '.', color='g', label="Residual")
        plt.xlabel('uvdist')
        plt.ylabel('Amp Visibility')
        plt.legend()
        plt.savefig('result/400_'+context+'.jpg')


    for context in contexts:

        targetimage_list = [create_image_from_visibility(vt, npixel=npixel, nchan=1,
                                                polarisation_frame=PolarisationFrame('stokesI'))]  #cellsize=6.8e-5

        vt_list = [vt]


        print('Processing context %s' % context)
        if context == 'wprojection':
            future = invert_list_serial_workflow(vt_list, targetimage_list, context=context, facets=4)  #, gcfcf=[gcfcf_clipped]

        elif context == 'facets':
            future = invert_list_serial_workflow(vt_list, targetimage_list, context=context, facets=4)

        # elif context == 'timeslice':
        #     future = invert_list_serial_workflow(vt_list, targetimage_list, context=context, vis_slices=vis_timeslices(vt, 'auto'))

        # elif context == 'wstack':
        #     future = invert_list_serial_workflow(vt_list, targetimage_list, context=context, vis_slices=31)

        else:
            future = invert_list_serial_workflow(vt_list, targetimage_list, context=context)

        # result = arlexecute.compute(future, sync=True)
        targetimage = future[0][0]

        show_image(targetimage)
        plt.title(context)
        plt.savefig('result/400_I_'+context+'.jpg')

        #plt.show()

        print("Dirty Image %s" % qa_image(targetimage, context="imaging-fits notebook, using processor %s" % context))

        export_image_to_fits(targetimage, '%s/imaging-fits_dirty_%s.fits' % ('result', context))
        comps = find_skycomponents(targetimage, fwhm=1.0, threshold=10.0, npixels=5)

        plt.clf()
        for comp in comps:
            distance = comp.direction.separation(model.phasecentre)
            dft_flux = sum_visibility(vt, comp.direction)[0]
            err = (comp.flux[0, 0] - dft_flux) / dft_flux
            plt.plot(distance, err, '.', color='r')
        plt.ylabel('Fractional error of image vs DFT')
        plt.xlabel('Distance from phasecentre (deg)')
        plt.title(
            "Fractional error in %s recovered flux vs distance from phasecentre" %
            context)
        plt.savefig('result/400_D_'+context+'.jpg')
        # plt.show()

        checkpositions = True
        if checkpositions:
            plt.clf()
            for i in range(len(comps)):
                ocomp, sep = find_nearest_skycomponent(comps[i].direction, original_comps)
                plt.plot(
                    (comps[i].direction.ra.value - ocomp.direction.ra.value) /
                    targetimage.wcs.wcs.cdelt[0],
                    (comps[i].direction.dec.value - ocomp.direction.dec.value) /
                    targetimage.wcs.wcs.cdelt[1],
                    '.',
                    color='r')

            plt.xlabel('delta RA (pixels)')
            plt.ylabel('delta DEC (pixels)')
            plt.title("%s: Position offsets" % context)
            plt.savefig('result/400_CK_'+context+'.jpg')

            # plt.show()
