# %matplotlib inline

import os
import sys
import numpy
import matplotlib
matplotlib.use('Agg')

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (6.0, 6.0)
pylab.rcParams['image.cmap'] = 'rainbow'

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join('..', '..'))

from data_models.parameters import arl_path


from matplotlib import pylab

import numpy

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_models.polarisation import PolarisationFrame
from astropy.coordinates import EarthLocation, SkyCoord

from wrappers.serial.image.iterators import image_raster_iter
from processing_library.image.operations import create_w_term_like
from wrappers.serial.simulation.configurations import create_configuration_from_file
# Use serial wrappers by default
from wrappers.serial.visibility.base import create_visibility, create_visibility, create_visibility_from_rows
from wrappers.serial.skycomponent.operations import create_skycomponent
from wrappers.serial.image.operations import show_image, export_image_to_fits, qa_image, smooth_image
from wrappers.serial.visibility.iterators import vis_timeslice_iter
from wrappers.serial.simulation.configurations import create_named_configuration
from wrappers.serial.imaging.base import invert_2d, create_image_from_visibility, \
    predict_skycomponent_visibility, advise_wide_field

from wrappers.serial.visibility.iterators import vis_timeslice_iter
from wrappers.serial.imaging.weighting import weight_visibility
from wrappers.serial.visibility.iterators import vis_timeslices

from wrappers.arlexecute.griddata.kernels import create_awterm_convolutionfunction
from wrappers.arlexecute.griddata.convolution_functions import apply_bounding_box_convolutionfunction

# Use arlexecute for imaging
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

doplot = True

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
    if name=='MUSER-2':
        lowcore = create_configuration_from_file(antfile="muser-2.csv",
                                                mount='altaz', names='MUSER_%d',
                                                diameter=2.0, name='MUSER', location=location, **kwargs)
    else:
        lowcore = create_configuration_from_file(antfile="muser-1.csv",
                                                mount='altaz', names='MUSER_%d',
                                                diameter=4.0, name='MUSER', location=location, **kwargs)
    return lowcore

if __name__ == '__main__':

    fh = logging.FileHandler('musersim.log')
    fh.setLevel(logging.DEBUG)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(fh)

    pylab.rcParams['figure.figsize'] = (5.0, 5.0)
    pylab.rcParams['image.cmap'] = 'rainbow'
    pylab.rcParams['font.size'] = 9

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 9}
    matplotlib.rc('font', **font)

    test_list = (('MUSER-1',1400),) #,('MUSER-1',1400),('MUSER-2',2000),('MUSER-2',15000))

    arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=16 * 1024 * 1024 * 1024, n_workers=8,
                          local_dir=dask_dir)

    for (muser,freq) in test_list:
        lowcore = create_configuration(muser)

        # lowcore = create_named_configuration('MUSER')
        # arlexecute.set_client(use_dask=True)

        times = numpy.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * (numpy.pi / 12.0)  #[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        # times = numpy.array([0.0]) * (numpy.pi / 12.0)  #[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        frequency = numpy.array([freq*1e6])
        # Directory of storing result

        results_dir = str(freq)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        channel_bandwidth = numpy.array([25e6])
        reffrequency = numpy.max(frequency)
        phasecentre = SkyCoord(ra=+15 * u.deg, dec= 45 * u.deg, frame='icrs', equinox='J2000')

        # plt.show()

        # vt.data['vis'] *= 0.0
        if freq==400:
            npixel=256
        elif freq>1000 and freq<2000:
            npixel = 1024
        elif freq>=2000 and freq<6000:
            npixel = 2048
        elif freq >= 6000 and freq <= 12000:
            npixel = 4096
        else:
            npixel = 6144
        cellsize = 1. * numpy.pi / 180. / (npixel)
        # cellsize = None #1*3.1415926535/180./npixel
        facets = 8
        flux = numpy.array([[200.0]])

        vt = create_visibility(lowcore, times, frequency, channel_bandwidth=channel_bandwidth,
                            weight=1.0, phasecentre=phasecentre,
                            polarisation_frame=PolarisationFrame('stokesI'))

        advice = advise_wide_field(vt, wprojection_planes=1)

        # plt.clf()
        plt.plot(vt.data['uvw'][:, 0], vt.data['uvw'][:, 1], '.', color='b')
        plt.plot(-vt.data['uvw'][:, 0], -vt.data['uvw'][:, 1], '.', color='r')
        plt.xlabel('U (wavelengths)')
        plt.ylabel('V (wavelengths)')
        plt.title("UV coverage")
        plt.savefig(results_dir+'/UV_coverage.pdf',format='pdf')

        vt.data['vis'] *= 0.0

        model = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize=cellsize)
        spacing_pixels = npixel // facets
        log.info('Spacing in pixels = %s' % spacing_pixels)
        spacing = model.wcs.wcs.cdelt* spacing_pixels #180.0 * cellsize * spacing_pixels / numpy.pi
        centers = -3.0, -2.0, -1.0, -0.5, +0.5, +1.0, 2.0, 3.0
        comps = list()
        for iy in centers:
            for ix in centers:
                pra = int(round(npixel // 2 + ix * spacing_pixels - 1))
                pdec = int(round(npixel // 2 + iy * spacing_pixels - 1))
                sc = pixel_to_skycoord(pra, pdec, model.wcs)
                log.info("Component at (%f, %f) %s" % (pra, pdec, str(sc)))
                comp = create_skycomponent(flux=flux, frequency=frequency, direction=sc,
                                           polarisation_frame=PolarisationFrame("stokesI"))
                comps.append(comp)
        predict_skycomponent_visibility(vt, comps)

        arlexecute.set_client(use_dask=True)

        dirty = create_image_from_visibility(vt, npixel=npixel, cellsize=cellsize,
                                             polarisation_frame=PolarisationFrame("stokesI"))
        vt = weight_visibility(vt, dirty)

        future = invert_list_arlexecute_workflow([vt], [dirty], context='2d')
        dirty, sumwt = arlexecute.compute(future, sync=True)[0]

        if doplot:
            show_image(dirty,cm='hot')
            plt.title("Dirty image")
            plt.savefig('%s/%s_dirty.pdf' % (results_dir, str(freq)))

        print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirty.data.max(), dirty.data.min(), sumwt))

        export_image_to_fits(dirty, '%s/imaging-wterm_dirty.fits' % (results_dir))

        dirtyFacet = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize=cellsize)
        future = invert_list_arlexecute_workflow([vt], [dirtyFacet], facets=4, context='facets')
        dirtyFacet, sumwt = arlexecute.compute(future, sync=True)[0]
        dirtyFacet = smooth_image(dirtyFacet)
        if doplot:
            show_image(dirtyFacet,cm='hot')
            plt.title("Smoothed model image")
            plt.savefig('%s/%s_smooth.pdf' % (results_dir, str(freq)))

        print(
            "Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirtyFacet.data.max(), dirtyFacet.data.min(), sumwt))
        export_image_to_fits(dirtyFacet, '%s/imaging-wterm_dirtyFacet.fits' % (results_dir))

        dirtyFacet2 = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize=cellsize)
        future = invert_list_arlexecute_workflow([vt], [dirtyFacet2], facets=2, context='facets')
        dirtyFacet2, sumwt = arlexecute.compute(future, sync=True)[0]

        if doplot:
            show_image(dirtyFacet2,cm='hot')
            plt.title("Dirty Facet image")
            plt.savefig('%s/%s_dirtyfacet2.pdf' % (results_dir, str(freq)))

        print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (
        dirtyFacet2.data.max(), dirtyFacet2.data.min(), sumwt))
        export_image_to_fits(dirtyFacet2, '%s/imaging-wterm_dirtyFacet2.fits' % (results_dir))

        if doplot:
            wterm = create_w_term_like(model, phasecentre=vt.phasecentre, w=numpy.max(vt.w))
            show_image(wterm)
            plt.title("Wterm image")
            plt.savefig('%s/%s_wterm.pdf' % (results_dir, str(freq)))

        dirtywstack = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize=cellsize)
        future = invert_list_arlexecute_workflow([vt], [dirtywstack], vis_slices=101, context='wstack')
        dirtywstack, sumwt = arlexecute.compute(future, sync=True)[0]

        show_image(dirtywstack)
        plt.title("dirtywstack image")
        plt.savefig('%s/%s_dirtywstack.pdf' % (results_dir, str(freq)))
        # plt.show()

        print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
              (dirtywstack.data.max(), dirtywstack.data.min(), sumwt))

        export_image_to_fits(dirtywstack, '%s/imaging-wterm_dirty_wstack.fits' % (results_dir))

        for rows in vis_timeslice_iter(vt):
            visslice = create_visibility_from_rows(vt, rows)
            dirtySnapshot = create_image_from_visibility(visslice, npixel=npixel, npol=1, cellsize=cellsize,
                                                         compress_factor=0.0)
            future = invert_list_arlexecute_workflow([visslice], [dirtySnapshot], context='2d')
            dirtySnapshot, sumwt = arlexecute.compute(future, sync=True)[0]
            dirtySnapshot = smooth_image(dirtySnapshot)
            print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
                  (dirtySnapshot.data.max(), dirtySnapshot.data.min(), sumwt))
            if doplot:

                dirtySnapshot.data -= dirtyFacet.data
                show_image(dirtySnapshot,cm='hot')
                plt.title("Hour angle %.2f hours" % (numpy.average(visslice.time) * 12.0 / 43200.0))
                plt.savefig('%s/%s_snapshot%03d.pdf' % (results_dir, str(freq),(numpy.average(visslice.time) * 12.0 / 43200.0)))
                # plt.show()

        dirtyTimeslice = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize=cellsize)
        future = invert_list_arlexecute_workflow([vt], [dirtyTimeslice], vis_slices=vis_timeslices(vt, 'auto'),
                                               padding=2, context='2d')
        dirtyTimeslice, sumwt = arlexecute.compute(future, sync=True)[0]

        dirtyTimeslice.data -= dirtyFacet.data
        show_image(dirtyTimeslice)
        plt.title("Dirty Timeslice")
        plt.savefig(
            '%s/%s_snapshot.pdf' % (results_dir, str(freq)))

        # plt.show()

        print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" %
              (dirtyTimeslice.data.max(), dirtyTimeslice.data.min(), sumwt))

        export_image_to_fits(dirtyTimeslice, '%s/imaging-wterm_dirty_Timeslice.fits' % (results_dir))

        # dirtyWProjection = create_image_from_visibility(vt, npixel=npixel, npol=1, cellsize= cellsize)
        #
        # gcfcf = create_awterm_convolutionfunction(model, nw=4100, wstep=4100.0 / 4, oversampling=8,
        #                                           support=60,
        #                                           use_aaf=True)
        #
        # future = invert_list_arlexecute_workflow([vt], [dirtyWProjection], context='2d', gcfcf=[gcfcf])
        #
        # dirtyWProjection, sumwt = arlexecute.compute(future, sync=True)[0]
        #
        # if doplot:
        #     show_image(dirtyWProjection)
        #     plt.title("Dirty Wprojection")
        #     plt.savefig(
        #         '%s/%s_dirtywprojectionsnapshot.pdf' % (results_dir, str(freq)))
        #
        # print("Max, min in dirty image = %.6f, %.6f, sumwt = %f" % (dirtyWProjection.data.max(),
        #                                                             dirtyWProjection.data.min(), sumwt))
        # export_image_to_fits(dirtyWProjection, '%s/imaging-wterm_dirty_WProjection.fits' % (results_dir))