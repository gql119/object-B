"""Instance-local risk injection, reusing unchanged original Q/map/cache code.

No module global is mutated. FunctionType preserves the original code object;
only two explicit dependency bindings differ inside this isolated cache type.
The ordinary VOC/native cache remains the original class and globals.
"""
from types import FunctionType
from ue_framework.methods.s4_tfss import CleanMapCache
from ue_framework.methods.tfss_clean_maps import build_maps


class RiskCleanMapCache(CleanMapCache):
    @staticmethod
    def _build_maps(trainer,items):
        estimator=getattr(trainer,'nt_risk_estimator',None)
        if estimator is None:raise ValueError('Independent-risk cache requires its bound explicit estimator')
        namespace=dict(build_maps.__globals__)
        namespace['gradients']=estimator
        function=FunctionType(build_maps.__code__,namespace,build_maps.__name__,build_maps.__defaults__,build_maps.__closure__)
        assert function.__code__ is build_maps.__code__
        return function(trainer,items)

    def get(self,trainer,items,paths):
        namespace=dict(CleanMapCache.get.__globals__)
        namespace['build_maps']=self._build_maps
        function=FunctionType(CleanMapCache.get.__code__,namespace,CleanMapCache.get.__name__,
                              CleanMapCache.get.__defaults__,CleanMapCache.get.__closure__)
        assert function.__code__ is CleanMapCache.get.__code__
        return function(self,trainer,items,paths)


def cache_class(manifest):
    from .risk_fallback import POLICY
    policy=manifest.get('risk_estimator_policy')
    if policy is None:return CleanMapCache
    if policy!=POLICY:raise ValueError('Unknown independent risk-cache policy')
    return RiskCleanMapCache
