
class cluster_manager:

    def __init__(self, cluster_info):
        self.cluster = cluster_info
        self.hosts = [c['ip'] for c in cluster_info]
        self.ips = [h.split(':')[0] for h in self.hosts]
        self.names = [c['name'] for c in cluster_info]
        self.types = [c['type'] for c in cluster_info]
        self.tasks = [c['task'] for c in cluster_info]

    def getIndex(self, attr, str):
        return getattr(self, attr).index(str)

    def get(self, attr):
        return getattr(self, attr)

    def match_index(self, layers):
        for layer in layers:
            layer['device'] = self.getIndex('names', layer['device'])