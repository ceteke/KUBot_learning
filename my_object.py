class MyObject():

    def __init__(self,name,pose):
        self.name = name
        self.pose = pose
        self.id = '%s%d' % (name,pose)
        self.X = []
        self.y = []

    def __str__(self):
        return self.id
