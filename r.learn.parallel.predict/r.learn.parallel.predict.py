#!/usr/bin/env python3
############################################################################
#
# MODULE:       r.learn.parallel.predict
# AUTHOR(S):    Anika Weinmann
# PURPOSE:      Applies the classification model parallel using r.learn.predict
# COPYRIGHT:    (C) 2020-2022 by mundialis GmbH & Co. KG and the GRASS
#               Development Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
############################################################################

# %module
# % description: Applies a classification model in parallel using r.learn.predict.
# % keyword: raster
# % keyword: classification
# % keyword: regression
# % keyword: machine learning
# % keyword: scikit-learn
# % keyword: prediction
# % keyword: parallel
# %end
# %flag
# % key: p
# % label: Output class membership probabilities
# % description: A raster layer is created for each class. For the case of a binary classification, only the positive (maximum) class is output
# % guisection: Optional
# %end
# %flag
# % key: z
# % label: Only predict class probabilities
# % guisection: Optional
# %end
# %flag
# % key: v
# % label: Create a VRT (Virtual Raster) as output
# % guisection: Optional
# %end

# %option G_OPT_I_GROUP
# % key: group
# % label: Group of raster layers used for prediction
# % description: GRASS imagery group of raster maps representing feature variables to be used in the machine learning model
# % required: yes
# % multiple: no
# %end

# %option G_OPT_F_INPUT
# % key: load_model
# % label: Load model from file
# % description: File representing pickled scikit-learn estimator model
# % required: yes
# % guisection: Required
# %end

# %option G_OPT_R_OUTPUT
# % key: output
# % label: Output Map
# % description: Raster layer name to store result from classification or regression model. The name will also used as a perfix if class probabilities or intermediate of cross-validation results are ordered as maps.
# % guisection: Required
# % required: yes
# %end

# %option
# % key: chunksize
# % type: integer
# % label: Number of pixels to pass to the prediction method
# % description: Number of pixels to pass to the prediction method. GRASS GIS reads raster by-row so chunksize is rounded down based on the number of columns
# % answer: 100000
# % guisection: Optional
# %end

# %option G_OPT_M_NPROCS
# % label: Number of parallel processes used for band importing in sen2cor
# % description: Number of cores for multiprocessing, -2 is n_cores-1
# % answer: -2
# % guisection: Optional
# %end

# %option
# % key: grid
# % type: integer
# % required: no
# % multiple: no
# % key_desc: rows,columns
# % description: Number of rows and columns in grid
# %end

import atexit
import sys
import os
import multiprocessing as mp

import grass.script as grass
from grass.pygrass.modules import Module, ParallelModuleQueue

# initialize global vars
rm_regions = []
rm_vectors = []
rm_rasters = []


def cleanup():
    nuldev = open(os.devnull, "w")
    kwargs = {"flags": "f", "quiet": True, "stderr": nuldev}
    for rmr in rm_regions:
        if rmr in [x for x in grass.parse_command("g.list", type="region")]:
            grass.run_command("g.remove", type="region", name=rmr, **kwargs)
    for rmv in rm_vectors:
        if grass.find_file(name=rmv, element="vector")["file"]:
            grass.run_command("g.remove", type="vector", name=rmv, **kwargs)
    for rmrast in rm_rasters:
        if grass.find_file(name=rmrast, element="raster")["file"]:
            grass.run_command("g.remove", type="raster", name=rmrast, **kwargs)


def set_test_nprocs(nprocs):
    # Test nprocs settings
    nprocs_real = mp.cpu_count()
    if nprocs == -2:
        procs = nprocs_real - 1
        grass.info("Using %d parallel processes" % (procs))
        return procs
    else:
        if nprocs > nprocs_real:
            grass.warning(
                "Using %d parallel processes but only %d CPUs available."
                % (nprocs, nprocs_real)
            )
        return nprocs


def main():

    global rm_regions, rm_rasters, rm_vectors

    # parallelization parameter
    n_jobs = set_test_nprocs(int(options["n_jobs"]))

    # parameter of r.learn.predict
    group = options["group"]
    output = options["output"]
    load_model = options["load_model"]
    chunksize = options["chunksize"]
    flags_str = ""
    for flag in flags:
        if flags[flag] and not flag == "v":
            flags_str += flag

    if options["grid"]:
        grid_rows_cols = options["grid"]
    else:
        grid_rows_cols = "%d,%d" % (n_jobs, n_jobs)

    # set some common environmental variables, like:
    os.environ.update(
        dict(
            GRASS_COMPRESS_NULLS="1",
            GRASS_COMPRESSOR="ZSTD",
            GRASS_MESSAGE_FORMAT="plain",
        )
    )

    # test if r.learn.predict is installed
    if not grass.find_program("r.learn.predict", "--help"):
        grass.fatal(
            _(
                "The 'r.learn.predict' module was not found, install it first:"
                + "\n"
                + "g.extension r.learn.ml2"
            )
        )

    if n_jobs > 1:
        grass.message(_("Generating grid to for parallelization ..."))
        grid = "tmp_grid_%s" % os.getpid()
        grass.run_command("v.mkgrid", map=grid, grid=grid_rows_cols)
        rm_vectors.append(grid)

        reg = grass.region()
        cats = list(
            grass.parse_command("v.category", input=grid, option="print").keys()
        )

        grass.message(_("Predict parallel on the grid cells ..."))
        # save current mapset
        env = grass.gisenv()
        # start_gisdbase = env['GISDBASE']
        # start_location = env['LOCATION_NAME']
        start_cur_mapset = env["MAPSET"]

        queue = ParallelModuleQueue(nprocs=n_jobs)
        classifications = []
        for cat in cats:
            new_mapset = "tmp_mapset_rlearnpredict_%s" % cat
            tmp_output = "%s_%s" % (output, cat)
            # Module
            r_grid_predict = Module(
                "r.learn.predict.worker",
                area=grid,
                where="cat=%s" % cat,
                mapset=new_mapset,
                nsres=reg["nsres"],
                ewres=reg["ewres"],
                group=group,
                output=tmp_output,
                load_model=load_model,
                run_=False,
                chunksize=chunksize,
            )
            classifications.append("%s@%s" % (tmp_output, new_mapset))
            queue.put(r_grid_predict)
        queue.wait()

        # verify that switchiing the mapset worked
        env = grass.gisenv()
        gisdbase = env["GISDBASE"]
        location = env["LOCATION_NAME"]
        cur_mapset = env["MAPSET"]
        if cur_mapset != start_cur_mapset:
            grass.fatal(
                "new mapset is %s, but should be %s" % (cur_mapset, start_cur_mapset)
            )

        for classification in classifications:
            name_mapset = classification.split("@")
            grass.run_command(
                "g.copy", raster="%s,%s" % (classification, name_mapset[0])
            )
            grass.utils.try_rmdir(os.path.join(gisdbase, location, name_mapset[1]))

        # patching
        grass.message(_("Patching the tiles ..."))
        grass.message(_("Current region for patching:\n%s") % grass.region())

        if len(classifications) > 1:
            module = "r.patch"
            if flags["v"]:
                module = "r.buildvrt"
            all_classified = [x.split("@")[0] for x in classifications]
            grass.run_command(module, input=all_classified, output=output)
            if not flags["v"]:
                rm_rasters.extend(all_classified)
        else:
            all_classified = [x.split("@")[0] for x in classifications][0]
            grass.run_command("g.copy", raster=all_classified + "," + output)

        grass.message(_("Patching the tiles done"))
    else:
        grass.run_command(
            "r.learn.predict",
            group=group,
            output=output,
            load_model=load_model,
            chunksize=chunksize,
            flags=flags_str,
        )
    return 0


if __name__ == "__main__":
    options, flags = grass.parser()
    atexit.register(cleanup)
    sys.exit(main())
