# 全局默认配置文件
# 集中管理所有默认配置，避免在多个地方重复设置

# 默认模型配置
DEFAULT_ENABLED_MODELS = {
    'glove': True,
    'head': False
}

DEFAULT_MODEL_CONFIDENCE = {
    'glove': 0.85,
    'head': 0.7
}

DEFAULT_MODEL_THRESHOLDS = {
    'glove': 2,
    'head': 10
}
