"""Class-conditioned Refinement Module (CRM).

Class-conditioned refinement head. Learnable per-class shape templates and a
topology network refine the raw segmentation logits; a learnable residual gate
blends the raw and refined logits. The two networks share one CRM head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassConditionedRefinement(nn.Module):
    """Class-conditioned Refinement Module (CRM)"""
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Learned shape templates
        self.shape_templates = nn.Parameter(torch.randn(num_classes, 1, 64, 64))
        
        # Shape matching network
        self.shape_matcher = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Topology preserving network
        self.topology_net = nn.Sequential(
            nn.Conv2d(num_classes+1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 3, padding=1)
        )
        self.weights=nn.Parameter(torch.ones(2),requires_grad=True)
        
        
    def forward(self, predictions):
        B, C, H, W = predictions.shape
        
        # Resize templates to match prediction size
        templates = F.interpolate(self.shape_templates, size=(H, W), mode='bilinear')
        
        # Shape matching
        shape_similarity = self.shape_matcher(predictions)
        shape_prior = (shape_similarity.softmax(dim=1).unsqueeze(2) * templates.unsqueeze(0)).sum(dim=1)
        
        # Topology preservation
        topology_preserved = self.topology_net(torch.cat([predictions,shape_prior],dim=1))
        
        # Combine
        weights=torch.softmax(self.weights,dim=0)
        refined = weights[0]*predictions +  weights[1]*topology_preserved
        
        return refined

# Alias: the refinement head is referred to as CRM in some places.
CRM = ClassConditionedRefinement
