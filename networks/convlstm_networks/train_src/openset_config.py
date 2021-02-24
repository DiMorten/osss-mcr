class LocoClass():
    pass

class SpecifyKnownClasses(LocoClass):
    def setLabelWithLocoClass(self, label_with_loco_class, loco_class):
        label_with_loco_class[label_with_loco_class == loco_class] = 0
        return label_with_loco_class

class SpecifyUnknownClasses(LocoClass):
    def setLabelWithLocoClass(self, label_with_loco_class, loco_class):
        label_with_loco_class[label_with_loco_class == loco_class] = 0
        return label_with_loco_class
