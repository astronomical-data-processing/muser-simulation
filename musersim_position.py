# %matplotlib inline

import os
import sys
import numpy
import matplotlib
from wrappers.serial.simulation.configurations import create_named_configuration
import argparse

matplotlib.use('Agg')

from matplotlib import pylab

from matplotlib import pyplot as plt
# from matplotlib import plt.savefig
from astropy.coordinates import EarthLocation, SkyCoord

pylab.rcParams['figure.figsize'] = (6.0, 6.0)
pylab.rcParams['image.cmap'] = 'rainbow'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
from wrappers.serial.imaging.weighting import weight_visibility
from wrappers.arlexecute.griddata.kernels import create_awterm_convolutionfunction
from wrappers.arlexecute.griddata.convolution_functions import apply_bounding_box_convolutionfunction

# Use workflows for imaging
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from workflows.shared.imaging.imaging_shared import imaging_contexts
from workflows.arlexecute.imaging.imaging_arlexecute import predict_list_arlexecute_workflow, \
    invert_list_arlexecute_workflow, deconvolve_list_arlexecute_workflow, \
    residual_list_arlexecute_workflow, restore_list_arlexecute_workflow

import logging
from data_models.parameters import arl_path

dask_dir = BASE_DIR + "/dask-work-space/"  # arl_path('test_results/dask-work-space')
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
    if name == 'MUSER-2':
        lowcore = create_configuration_from_file(antfile="muser-2.csv",
                                                 mount='altaz', names='MUSER_%d',
                                                 diameter=2.0, name='MUSER', location=location, **kwargs)
    else:
        lowcore = create_configuration_from_file(antfile="muser-1.csv",
                                                 mount='altaz', names='MUSER_%d',
                                                 diameter=4.0, name='MUSER', location=location, **kwargs)
    return lowcore


if __name__ == '__main__':

    if len(sys.argv) ==1:
        muser='MUSER-1'
    else:
        muser='MUSER-2'

    fh = logging.FileHandler('musersim.log')
    fh.setLevel(logging.DEBUG)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(fh)

    pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    pylab.rcParams['image.cmap'] = 'rainbow'
    pylab.rcParams['font.size'] = 10

    # Define Font for Matplotlib
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    matplotlib.rc('font', **font)

    # MUSER-1 400Mhz - 1975 Mhz
    # MUSER-2 2000 Mhz - 15000 Mhz

    test_list = (('MUSER-1', 400),) #, ('MUSER-1', 1400))  # ,('MUSER-2',2000),('MUSER-2',15000))
    resultx = []
    resultxrms = []
    resulty = []
    resultyrms = []
    disfreq = []
    discomp= []
    if muser =='MUSER-1':
        start = 400
        end = 2000
        step = 100
    else:
        start = 2000
        end = 6000
        step = 200
    for freq in range(start, end, step):        # storedir = str(freq)
        # if not os.path.exists(storedir):
        #     os.makedirs(storedir)

        lowcore = create_configuration(muser)

        # lowcore = create_named_configuration('LOWBD2-CORE')

        # lowcore = create_named_configuration('MUSER')
        # arlexecute.set_client(use_dask=True)
        arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=8 * 1024 * 1024 * 1024, n_workers=8,
                              local_dir=dask_dir)

        if freq>=400 and freq<600:
            npixel=256
        elif freq>=600 and freq<1200:
            npixel = 512
        elif freq>=1200 and freq<2000:
            npixel = 1024
        elif freq>=2000 and freq<4000:
            npixel = 2048
        elif freq >= 4000 and freq < 8000:
            npixel = 4096
        else:
            npixel = 4096
        cellsize = 1. * numpy.pi / 180. / (npixel)

        times = numpy.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * (
                    numpy.pi / 12.0)  # [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        # times = numpy.array([0.0]) * (numpy.pi / 12.0)  #[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        frequency = numpy.array([freq * 1e6])
        # Directory of storing result

        channel_bandwidth = numpy.array([25e6])
        reffrequency = numpy.max(frequency)
        phasecentre = SkyCoord(ra=+80 * u.deg, dec=41 * u.deg, frame='icrs', equinox='J2000')
        vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                               weight=1.0, phasecentre=phasecentre,
                               polarisation_frame=PolarisationFrame('stokesI'))

        # print("##################", vt.data)
        advice = advise_wide_field(vt, wprojection_planes=4)

        # plt.clf()
        # plt.plot(vt.data['uvw'][:, 0], vt.data['uvw'][:, 1], '.', color='b')
        # plt.plot(-vt.data['uvw'][:, 0], -vt.data['uvw'][:, 1], '.', color='r')
        # plt.xlabel('U (wavelengths)')
        # plt.ylabel('V (wavelengths)')
        # plt.title("UV coverage")
        # plt.savefig(storedir + '/UV_coverage.pdf', format='pdf')
        # plt.show()

        vt.data['vis'] *= 0.0

        # print("*****************", vt.data['vis'])

        model = create_image_from_visibility(vt, npixel=npixel, nchan=1, cellsize=cellsize,
                                             polarisation_frame=PolarisationFrame('stokesI'))
        centre = model.wcs.wcs.crpix - 1
        spacing_pixels = npixel // 8
        # For an observation for the Sun, the disk is about 34 arcmin, if total FOV is 60 arcmin, then
        #   the disk edge would be 34/60*8
        log.info('Spacing in pixels = %s' % spacing_pixels)
        spacing = model.wcs.wcs.cdelt * spacing_pixels
        # locations = [-3.5, -2.5, -2.27, -1.5, -1., -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
        locations = [-3.5, -3.0, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.0, 3.5]

        original_comps = []
        # We calculate the source positions in pixels and then calculate the
        # world coordinates to put in the skycomponent description
        for iy in locations:
            for ix in locations:
                if True:  # ix >= iy:
                    p = int(round(centre[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
                        int(round(centre[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
                    sc = pixel_to_skycoord(p[0], p[1], model.wcs)
                    # log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))
                    flux = numpy.array([[200.0]])  # numpy.array([[100.0 + 2.0 * ix + iy * 20.0]])
                    comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                               polarisation_frame=PolarisationFrame('stokesI'))
                    original_comps.append(comp)
                    insert_skycomponent(model, comp)

        predict_skycomponent_visibility(vt, original_comps)

        # cmodel = smooth_image(model)
        # show_image(cmodel)
        # plt.title("Smoothed model image")
        # plt.savefig('%s/%s_smoothed.pdf' % (storedir, str(freq)))
        # plt.show()

        arlexecute.set_client(use_dask=True)

        dirty = create_image_from_visibility(vt, npixel=npixel, cellsize=cellsize,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        vt = weight_visibility(vt, dirty)

        future = invert_list_arlexecute_workflow([vt], [dirty], context='2d')
        dirty, sumwt = arlexecute.compute(future, sync=True)[0]

        cmodel = smooth_image(dirty)
        show_image(cmodel)
        plt.title("Smoothed model image")
        plt.savefig('%s/%s_smoothed.pdf' % ('./', str(freq)))
        plt.show()

        comps = find_skycomponents(cmodel, fwhm=1.0, threshold=500.0, npixels=npixel//60)
        plt.clf()
        log.info('Total components = %s' % len(comps))
        discomp.append(len(comps))
        x_offset=[]
        y_offset=[]
        offset=[]

        for i in range(len(comps)):
            ocomp, sep = find_nearest_skycomponent(comps[i].direction, original_comps)
            # log.info('Position offset %d %f %f' % (i, comps[i].direction.ra.value - ocomp.direction.ra.value,
            #                                        comps[i].direction.dec.value - ocomp.direction.dec.value))
            x_offset.append(numpy.abs((comps[i].direction.ra.value - ocomp.direction.ra.value)))
            y_offset.append(numpy.abs((comps[i].direction.dec.value - ocomp.direction.dec.value)))
            offset.append(numpy.sqrt((comps[i].direction.ra.value - ocomp.direction.ra.value)**2+(comps[i].direction.dec.value - ocomp.direction.dec.value)**2))

            # plt.plot((comps[i].direction.ra.value - ocomp.direction.ra.value),
            #          (comps[i].direction.dec.value - ocomp.direction.dec.value),
            #          '.', color='r')
            # plt.plot((comps[i].direction.ra.value - ocomp.direction.ra.value) / cmodel.wcs.wcs.cdelt[0],
            #          (comps[i].direction.dec.value - ocomp.direction.dec.value) / cmodel.wcs.wcs.cdelt[1],
            #          '.', color='r')
        xoffset = numpy.array(x_offset)
        yoffset = numpy.array(y_offset)
        offset = numpy.array(offset)
        disfreq.append(freq)
        resultx.append( numpy.mean(xoffset))# , numpy.std(xoffset)) )
        resultxrms.append(numpy.std(xoffset))
        resulty.append( numpy.mean(yoffset))# , numpy.std(xoffset)) )
        resultyrms.append(numpy.std(yoffset))

        # log.info('Position offset x: %f %f %f %f' % ( xoffset.min(), xoffset.max(),numpy.mean(xoffset), numpy.std(xoffset)))
        # plt.xlabel('delta RA (pixels)')
        # plt.ylabel('delta DEC (pixels)')
        # plt.title("Recovered - Original position offsets")
        # plt.savefig('%s/%s_R_O.pdf' % (storedir, str(freq)))
        # plt.show()
    for i in range(len(resultx)):
        log.info('offset Freq: %d: %f %f %f %f' % (disfreq[i],resultx[i],resultxrms[i],resulty[i],resultyrms[i]))

    plt.figure(figsize=(5,3))
    plt.clf()
    plt.plot(disfreq, resultx, '.', color='b')
    # plt.plot(disfreq, discomp, '*', color='r')
    plt.xlabel('Frequency)')
    plt.ylabel('RA Offset')
    plt.savefig('%s/%s_ra_offset.pdf' % (BASE_DIR, muser))

    plt.figure(figsize=(5,3))
    plt.clf()
    plt.plot(disfreq, resulty, '.', color='b')
    # plt.plot(disfreq, discomp, '*', color='r')
    plt.xlabel('Frequency)')
    plt.ylabel('DEC Offset')
    plt.savefig('%s/%s_dec_offset.pdf' % (BASE_DIR, muser))

    # plt.title("Recovered - Original position offsets")
    # plt.show()

