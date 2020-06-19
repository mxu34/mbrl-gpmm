
from .HalfCheetahEnv_template import HalfCheetahEnv

class HalfCheetahSlope00m34(HalfCheetahEnv):
    def __init__(self):
        super().__init__(model_xml_path='half_cheetah00m34.xml')
