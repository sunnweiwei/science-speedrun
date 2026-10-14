"""Independent parity against pinned WeatherBench-X on nontrivial random panels."""
import json
import numpy as np
import xarray as xr
from weatherbenchX.metrics.probabilistic import RelativeEconomicValue
from rev import confusion,potential_value

rng=np.random.default_rng(739)
ensemble=rng.uniform(size=(3,56,19))
truth=rng.uniform(size=(3,19))
weights=rng.uniform(.1,1,size=19)
threshold=np.full(19,.7)
costs=np.geomspace(.001,.7,31)
ours=potential_value(np.mean([confusion(ensemble[i],truth[i],threshold,weights) for i in range(3)],axis=0),costs)
probabilities=xr.DataArray((ensemble>.7).mean(axis=1),dims=['case','point'])
observed=xr.DataArray(truth>.7,dims=['case','point'])
area=xr.DataArray(weights,dims=['point'])
metric=RelativeEconomicValue(ensemble_size=56,cost_loss_ratios=costs)
statistics={name:{'field':stat.compute({'field':probabilities},{'field':observed})['field'].weighted(area).mean('point').mean('case')}
            for name,stat in metric.statistics.items()}
reference=metric.values_from_mean_statistics(statistics)['field'].max('threshold').values
np.testing.assert_allclose(ours,reference,rtol=1e-12,atol=1e-12)
print(json.dumps(dict(status='passed',cases=3,members=56,cost_loss_ratios=31,max_absolute_difference=float(np.max(np.abs(ours-reference))))))
