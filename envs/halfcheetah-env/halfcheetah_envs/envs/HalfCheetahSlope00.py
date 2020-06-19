
from .HalfCheetahEnv_template import HalfCheetahEnv

class HalfCheetahSlope00(HalfCheetahEnv):
    def __init__(self):
        super().__init__(model_xml_path='half_cheetah00.xml')
