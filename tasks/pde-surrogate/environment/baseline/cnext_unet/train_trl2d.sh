#!/usr/bin/env bash
# Convenience: the turbulent-layer run with the published settings (200 epochs, 12 h cap). See train_dataset.sh.
exec bash "$(dirname "$(readlink -f "$0")")/train_dataset.sh" turbulent_radiative_layer_2D turbulent_radiative_layer_2D 200
