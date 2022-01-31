#!/usr/bin/env python3
############################################################################
#
# MODULE:       r.learn.predict.worker
# AUTHOR(S):    Anika Weinmann
# PURPOSE:      Applies classification model to a region.
#               It is called in parallel by r.learn.parallel.predict.
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
# % description: Applies a classification model to a region. It is called in parallel by r.learn.parallel.predict.
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

# %option
# % key: mapset
# % type: string
# % required: yes
# % multiple: no
# % key_desc: name
# % label: Name of mapset to create
# % guisection: Required
# %end

# %option G_OPT_V_INPUT
# % key: area
# % label: Input vector map to set the region with a where condition
# % guisection: Required
# %end

# %option G_OPT_DB_WHERE
# % key: where
# % required: yes
# % multiple: no
# % label: Where condition to set the region to the area
# % guisection: Required
# %end

# %option
# % key: nsres
# % type: string
# % required: no
# % multiple: no
# % key_desc: value
# % description: North-south 2D grid resolution
# % guisection: Resolution
# %end
# %option
# % key: ewres
# % type: string
# % required: no
# % multiple: no
# % key_desc: value
# % description: East-west 2D grid resolution
# % guisection: Resolution
# %end


import sys
import os
import shutil

import grass.script as grass


def main():

    # parallelization parameter
    new_mapset = options["mapset"]
    area = options["area"]
    where = options["where"]
    nsres = options["nsres"]
    ewres = options["ewres"]

    # parameter of r.learn.predict
    group = options["group"]
    output = options["output"]
    load_model = options["load_model"]
    chunksize = options["chunksize"]
    flags_str = ""
    for flag in flags:
        if flags[flag]:
            flags_str += flag

    # set some common environmental variables, like:
    os.environ.update(
        dict(
            GRASS_COMPRESS_NULLS="1",
            GRASS_COMPRESSOR="ZSTD",
            GRASS_MESSAGE_FORMAT="plain",
        )
    )

    grass.message(_("Prediction of region %s, %s...") % (area, where))

    reg = grass.parse_command("v.db.select", flags="r", map=area, where=where)

    # actual mapset, location, ...
    env = grass.gisenv()
    gisdbase = env["GISDBASE"]
    location = env["LOCATION_NAME"]
    old_mapset = env["MAPSET"]

    grass.message("New mapset %s" % new_mapset)
    grass.utils.try_rmdir(os.path.join(gisdbase, location, new_mapset))

    gisrc = os.environ["GISRC"]
    newgisrc = "%s_%s" % (gisrc, str(os.getpid()))
    grass.try_remove(newgisrc)
    shutil.copyfile(gisrc, newgisrc)
    os.environ["GISRC"] = newgisrc

    grass.message("GISRC: %s" % os.environ["GISRC"])
    grass.run_command("g.mapset", flags="c", mapset=new_mapset)

    # verify that switching the mapset worked
    cur_mapset = grass.gisenv()["MAPSET"]
    if cur_mapset != new_mapset:
        grass.fatal("new mapset is %s, but should be %s" % (cur_mapset, new_mapset))

    grass.run_command("g.region", flags="a", nsres=nsres, ewres=ewres, grow=1, **reg)
    grass.message(_("current region (%s):\n%s") % (where, grass.region()))

    # copy group to current mapset
    grass.run_command("g.copy", group=group + "@" + old_mapset + "," + group)

    # classification
    grass.run_command(
        "r.learn.predict",
        group=group,
        output=output,
        load_model=load_model,
        chunksize=chunksize,
        flags=flags_str,
        quiet=True,
    )

    grass.message(_("Prediction of region %s, %s is done") % (area, where))
    grass.utils.try_remove(newgisrc)
    return 0


if __name__ == "__main__":
    options, flags = grass.parser()
    sys.exit(main())
