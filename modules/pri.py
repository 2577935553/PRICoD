"""Pixel-Region-Instance contrast (PRI).

Cross-network contrastive alignment at three granularities -- pixel, region
(learned super-pixels) and instance (class-averaged) -- computed between the
two co-training networks. ``forward`` returns the averaged PRI loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelRegionInstanceContrast(nn.Module):
    """Pixel-Region-Instance contrast (PRI) for semi-supervised learning"""
    def __init__(self, feature_dim, num_classes=4, temperature=1.0,pixel_proj=True):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Multi-level projectors
        self.pixel_projector = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//2, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim//2, feature_dim, 1)
        ) if pixel_proj else nn.Identity()
        
        self.region_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        
        self.instance_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )
        
        # Region extractor
        self.region_extractor = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),  # 16 regions
            nn.Softmax(dim=1)
        )
        self.activation_pixel={}
        self.activation_region={}
        self.activation_instance={}
    
    def forward(self, feat_l1, feat_l2, feat_u1, feat_u2,
                pred_l1, pred_l2, pred_u1, pred_u2,
                unc_l1=None, unc_l2=None,
                unc_u1=None, unc_u2=None):
        """
        Forward pass - processes all data together
        
        Args:
            feat_l1, feat_l2: features from labeled data (B, C, H, W)
            feat_u1, feat_u2: features from unlabeled data (B, C, H, W)
            pred_l1, pred_l2: predictions from labeled data (B, num_classes, H, W)
            pred_u1, pred_u2: predictions from unlabeled data (B, num_classes, H, W)
            unc_u1, unc_u2: uncertainty maps for unlabeled data (optional)
        
        Returns:
            total_loss: combined loss from all three levels
        """
        
        B,C,H,W=feat_l1.shape
        
        # Get labels
        labels_l1 = pred_l1.argmax(dim=1).unsqueeze(1)  # B,1, H, W
        labels_l2 = pred_l2.argmax(dim=1).unsqueeze(1)
        labels_u1 = pred_u1.argmax(dim=1).unsqueeze(1)
        labels_u2 = pred_u2.argmax(dim=1).unsqueeze(1)
        
        labels_l1_rs = F.interpolate(labels_l1.float(), size=(H, W), mode='nearest').squeeze(1)
        labels_l2_rs = F.interpolate(labels_l2.float(), size=(H, W), mode='nearest').squeeze(1)
        labels_u1_rs = F.interpolate(labels_u1.float(), size=(H, W), mode='nearest').squeeze(1)
        labels_u2_rs = F.interpolate(labels_u2.float(), size=(H, W), mode='nearest').squeeze(1)
        
        pred_l1_rs = F.interpolate(pred_l1, size=(H, W), mode='nearest')
        pred_l2_rs = F.interpolate(pred_l2, size=(H, W), mode='nearest')
        pred_u1_rs = F.interpolate(pred_u1, size=(H, W), mode='nearest')
        pred_u2_rs = F.interpolate(pred_u2, size=(H, W), mode='nearest')
        
        # Compute losses at three levels
        pixel_loss = self.pixel_contrast(
            feat_l1, feat_l2, feat_u1, feat_u2,
            labels_l1_rs, labels_l2_rs, labels_u1_rs, labels_u2_rs,
            unc_l1,unc_l2,
            unc_u1, unc_u2
        )
        
        region_loss = self.region_contrast(
            feat_l1, feat_l2, feat_u1, feat_u2,
            pred_l1_rs, pred_l2_rs, pred_u1_rs, pred_u2_rs,
        )
        
        instance_loss = self.instance_contrast(
            feat_l1, feat_l2, feat_u1, feat_u2,
            pred_l1_rs, pred_l2_rs, pred_u1_rs, pred_u2_rs,
        )
        # print('pixel_loss:',pixel_loss,'region_loss',region_loss,'instance_loss',instance_loss)
        
        return (pixel_loss + region_loss + instance_loss) / 3
    
    def posMask(self, pred1, pred2):
        """Create positive mask based on predictions"""
        # Handle 1D (B,), 2D (B, N) and 3D (B, H, W) inputs
        if pred1.dim() == 1:
            # For instance-level: (B,) -> (B, 1)
            pred1_flat = pred1.unsqueeze(1)
            pred2_flat = pred2.unsqueeze(1)
        elif pred1.dim() == 3:
            # For pixel-level: (B, H, W) -> (B, H*W)
            B, H, W = pred1.shape
            pred1_flat = pred1.reshape(B, H*W)
            pred2_flat = pred2.reshape(B, H*W)
        else:
            # For region-level: already (B, N)
            pred1_flat = pred1
            pred2_flat = pred2
        
        B = pred1_flat.shape[0]
        pred1_one_hot = F.one_hot(pred1_flat.long(), num_classes=self.num_classes).float() # B,HW,NW
        pred2_one_hot = F.one_hot(pred2_flat.long(), num_classes=self.num_classes).float()
        mask_p = torch.bmm(pred1_one_hot, pred2_one_hot.transpose(1, 2))
        return mask_p
    
    def negMask(self, pred1, pred2):
        """Create negative mask based on predictions"""
        return 1 - self.posMask(pred1, pred2)
    
    def contrastive_loss(self, feat, feat_pos, feat_neg1, feat_neg2, 
                        label_pred, label_pos, label_neg1, label_neg2,unc=None,title=None):
        """
        Compute contrastive loss
        feat: anchor features (B, C, N) where N can be HW, 16, or 1
        """
        N, C, HW = feat.shape
        
        # Compute logits with temperature
        logits1 = torch.bmm(feat.transpose(1, 2), feat_pos) / self.temperature
        logits2 = torch.bmm(feat.transpose(1, 2), feat_neg1) / self.temperature
        logits3 = torch.bmm(feat.transpose(1, 2), feat_neg2) / self.temperature
            
        
        
        # if unc is not None:
        #     print(label_pred.shape)
        #     label_pred = label_pred * unc.unsqueeze(1)
            
        #     label_pos = label_pos * unc.unsqueeze(1)
        
        # Create masks
        mask_p = self.posMask(label_pred, label_pos).detach()
        mask_p2 = self.posMask(label_pred, label_neg1).detach()
        mask_p3 = self.posMask(label_pred, label_neg2).detach()
        if title=='pixel':
            self.activation_pixel['logits_l1_l2']=logits1
            self.activation_pixel['logits_l1_u1']=logits2
            self.activation_pixel['logits_l1_u2']=logits3
            self.activation_pixel['mask_l1_l2']=mask_p
            self.activation_pixel['mask_l1_u1']=mask_p2
            self.activation_pixel['mask_l1_u2']=mask_p3
        elif title=='region':
            self.activation_region['logits_l1_l2']=logits1
            self.activation_region['logits_l1_u1']=logits2
            self.activation_region['logits_l1_u2']=logits3
            self.activation_region['mask_l1_l2']=mask_p
            self.activation_region['mask_l1_u1']=mask_p2
            self.activation_region['mask_l1_u2']=mask_p3
        
        mask_n = self.negMask(label_pred, label_pos).detach()
        mask_n2 = self.negMask(label_pred, label_neg1).detach()
        mask_n3 = self.negMask(label_pred, label_neg2).detach()
        
        # Apply masks
        masked_p = mask_p * logits1
        masked_p2 = mask_p2 * logits2
        masked_p3 = mask_p3 * logits3
        
        masked_n = mask_n * logits1
        masked_n2 = mask_n2 * logits2
        masked_n3 = mask_n3 * logits3
        
        # Compute positive and negative scores
        P = torch.sum(masked_p + masked_p2 + masked_p3, dim=-1) / (torch.sum(mask_p + mask_p2 + mask_p3, dim=-1) + 1e-6)
        N = (torch.sum(masked_n, dim=-1) / (torch.sum(mask_n, dim=-1) + 1e-6) + 
             torch.sum(masked_n2, dim=-1) / (torch.sum(mask_n2, dim=-1) + 1e-6) + 
             torch.sum(masked_n3, dim=-1) / (torch.sum(mask_n3, dim=-1) + 1e-6))
        
        P_exp = torch.exp(P)
        N_exp = torch.exp(N)
        loss = -torch.mean(torch.log(P_exp / (P_exp + N_exp) + 1e-6))
        
        return loss
    
    def pixel_contrast(self, feat_l1, feat_l2, feat_u1, feat_u2,
                      labels_l1, labels_l2, labels_u1, labels_u2,unc_l1=None, unc_l2=None,
                      unc_u1=None, unc_u2=None):
        """Pixel-level contrastive learning"""
        # Project and normalize
        proj_l1 = F.normalize(self.pixel_projector(feat_l1), dim=1)
        proj_l2 = F.normalize(self.pixel_projector(feat_l2), dim=1)
        proj_u1 = F.normalize(self.pixel_projector(feat_u1), dim=1)
        proj_u2 = F.normalize(self.pixel_projector(feat_u2), dim=1)
        
        B, C, H, W = proj_l1.shape
        
        unc_l=None
        unc_u=None
        

        
        # # Apply uncertainty weighting to unlabeled features
        # if unc_u1 is not None and unc_l1 is not None:
        #     unc_l=1-(unc_l1+unc_l2)/2
        #     unc_u=1-(unc_u1+unc_u2)/2
        # else:
        #     unc_l=1-F.cosine_similarity(pred_l1,pred_l2).abs()
        #     unc_u=1-F.cosine_similarity(pred_u1,pred_u2).abs()
        
        # Reshape for contrastive computation
        proj_l1_flat = proj_l1.view(B, C, -1)  # B, C, HW
        proj_l2_flat = proj_l2.view(B, C, -1)
        proj_u1_flat = proj_u1.view(B, C, -1)
        proj_u2_flat = proj_u2.view(B, C, -1)
        
        # Compute 4 contrastive losses
        loss_l1 = self.contrastive_loss(
            proj_l1_flat, proj_l2_flat, proj_u1_flat, proj_u2_flat,
            labels_l1, labels_l2, labels_u1, labels_u2,unc_l,title='pixel'
        )
        
        loss_l2 = self.contrastive_loss(
            proj_l2_flat, proj_l1_flat, proj_u1_flat, proj_u2_flat,
            labels_l2, labels_l1, labels_u1, labels_u2,unc_l
        )
        
        loss_u1 = self.contrastive_loss(
            proj_u1_flat, proj_u2_flat, proj_l1_flat, proj_l2_flat,
            labels_u1, labels_u2, labels_l1, labels_l2,unc_u
        )
        
        loss_u2 = self.contrastive_loss(
            proj_u2_flat, proj_u1_flat, proj_l1_flat, proj_l2_flat,
            labels_u2, labels_u1, labels_l1, labels_l2,unc_u
        )
        
        return loss_l1 + loss_l2 + loss_u1 + loss_u2
    
    def pool_regions(self, feat, regions):
        """Pool features by regions"""
        B, C, _, _ = feat.shape
        feat_flat = feat.view(B, C, -1)  # B, C, HW
        regions_flat = regions.view(B, 16, -1)  # B, 16, HW
        return torch.bmm(feat_flat, regions_flat.transpose(1, 2))  # B, C, 16
    
    def project_regions(self, region_feat):
        """Project region features"""
        B, C, _ = region_feat.shape
        # B, C, 16 -> B, 16, C -> B*16, C
        region_feat_reshaped = region_feat.transpose(1, 2).contiguous().view(B * 16, C)
        proj = self.region_projector(region_feat_reshaped)  # B*16, C
        proj = proj.view(B, 16, -1).transpose(1, 2)  # B, C, 16
        return F.normalize(proj, dim=1)
    
    def get_region_labels(self, pred, regions):
        """Get region labels using vectorized operations"""
        B, _, H, W = pred.shape
        pred_labels = pred.argmax(dim=1)  # B, H, W
        regions_flat = regions.view(B, 16, -1)  # B, 16, HW
        pred_flat = pred_labels.view(B, -1, 1)  # B, HW, 1
        
        # Expand pred_flat for broadcasting
        pred_expanded = pred_flat.expand(B, -1, 16)  # B, HW, 16
        
        # Weight the labels by region membership
        weighted_labels = regions_flat.transpose(1, 2) * pred_expanded  # B, HW, 16
        
        # For each region, find the mode (most common label)
        # Using weighted voting
        region_labels = torch.zeros(B, 16, dtype=torch.long, device=pred.device)
        
        # Create one-hot encoding of labels
        num_pixels = H * W
        label_one_hot = F.one_hot(pred_labels.view(B, -1), num_classes=self.num_classes).float()  # B, HW, num_classes
        
        # Weight by region membership
        weighted_votes = torch.bmm(regions_flat, label_one_hot)  # B, 16, num_classes
        
        # Get the class with most votes for each region
        region_labels = weighted_votes.argmax(dim=-1)  # B, 16
        
        return region_labels
    
    def region_contrast(self, feat_l1, feat_l2, feat_u1, feat_u2,
                       pred_l1, pred_l2, pred_u1, pred_u2):
        """Region-level contrastive learning"""
        B, C, H, W = feat_l1.shape
        
        # Extract regions
        regions_l1 = self.region_extractor(feat_l1)  # B, 16, H, W
        regions_l2 = self.region_extractor(feat_l2)
        regions_u1 = self.region_extractor(feat_u1)
        regions_u2 = self.region_extractor(feat_u2)
        
        self.activation_region['regions_l1']=regions_l1
        self.activation_region['regions_l2']=regions_l2
        self.activation_region['regions_u1']=regions_u1
        self.activation_region['regions_u2']=regions_u2

     
        # Pool features by regions
        region_feat_l1 = self.pool_regions(feat_l1, regions_l1)
        region_feat_l2 = self.pool_regions(feat_l2, regions_l2)
        region_feat_u1 = self.pool_regions(feat_u1, regions_u1)
        region_feat_u2 = self.pool_regions(feat_u2, regions_u2)
        
        # Project and normalize
        proj_l1 = self.project_regions(region_feat_l1)
        proj_l2 = self.project_regions(region_feat_l2)
        proj_u1 = self.project_regions(region_feat_u1)
        proj_u2 = self.project_regions(region_feat_u2)
        self.activation_region['regions_feat_l1']=proj_l1
        self.activation_region['regions_feat_l2']=proj_l2
        self.activation_region['regions_feat_u1']=proj_u1
        self.activation_region['regions_feat_u2']=proj_u2 
        
        # Get region labels
        
        labels_l1 = self.get_region_labels(pred_l1, regions_l1)
        labels_l2 = self.get_region_labels(pred_l2, regions_l2)
        labels_u1 = self.get_region_labels(pred_u1, regions_u1)
        labels_u2 = self.get_region_labels(pred_u2, regions_u2)
        self.activation_region['regions_label_l1']=labels_l1
        self.activation_region['regions_label_l2']=labels_l2
        self.activation_region['regions_label_u1']=labels_u1
        self.activation_region['regions_label_u2']=labels_u2 
        
        # Compute contrastive losses
        loss_l1 = self.contrastive_loss(proj_l1, proj_l2, proj_u1, proj_u2,
                                       labels_l1, labels_l2, labels_u1, labels_u2,title='region')
        loss_l2 = self.contrastive_loss(proj_l2, proj_l1, proj_u1, proj_u2,
                                       labels_l2, labels_l1, labels_u1, labels_u2)
        loss_u1 = self.contrastive_loss(proj_u1, proj_u2, proj_l1, proj_l2,
                                       labels_u1, labels_u2, labels_l1, labels_l2)
        loss_u2 = self.contrastive_loss(proj_u2, proj_u1, proj_l1, proj_l2,
                                       labels_u2, labels_u1, labels_l1, labels_l2)
        
        return (loss_l1 + loss_l2 + loss_u1 + loss_u2)
    
    def instance_contrast(self, feat_l1, feat_l2, feat_u1, feat_u2,
                         pred_l1, pred_l2, pred_u1, pred_u2):
        features_all = torch.cat([feat_l1, feat_l2, feat_u1, feat_u2], dim=0)
        preds_all = torch.cat([pred_l1, pred_l2, pred_u1, pred_u2], dim=0)
        B, C, H, W = features_all.shape
        pred_labels = preds_all.argmax(dim=1) # (B, H, W)
        pred_one_hot = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float() # (B, K, H, W)
        features_flat = features_all.view(B, C, H * W) # (B, C, HW)
        pred_one_hot_flat = pred_one_hot.view(B, self.num_classes, H * W) # (B, K, HW)
        class_pixel_counts = pred_one_hot_flat.sum(dim=2) # (B, K)
        prototypes = torch.bmm(features_flat, pred_one_hot_flat.transpose(1, 2))
        prototypes = prototypes / (class_pixel_counts.unsqueeze(1) + 1e-8)
        prototypes_reshaped = prototypes.transpose(1, 2).contiguous().view(B * self.num_classes, C)# (B, C, K) -> (B, K, C) -> (B*K, C)
        self.activation_instance['instance_prototypes']=prototypes_reshaped 
        projected_prototypes = self.instance_projector(prototypes_reshaped)
        projected_prototypes = F.normalize(projected_prototypes, dim=1) # (B*K, C_proj)
        prototype_labels = torch.arange(self.num_classes).repeat(B).to(features_all.device) #(B*K,1)
        B_l = feat_l1.shape[0]
        B_u = feat_u1.shape[0]

        total_instance_ids = torch.cat([
            torch.arange(B_l),                
            torch.arange(B_l),                
            torch.arange(B_u) + B_l,          
            torch.arange(B_u) + B_l           
        ], dim=0).to(features_all.device)
        

        prototype_instance_ids = total_instance_ids.repeat_interleave(self.num_classes)
        self.activation_instance['instance_ids']=prototype_instance_ids 
        valid_mask = (prototype_labels != 0) & (class_pixel_counts.view(-1) > 0)
        if valid_mask.sum() < 2: 
            return torch.tensor(0.0, device=features_all.device)
        final_prototypes = projected_prototypes[valid_mask]
        final_labels = prototype_labels[valid_mask]
        final_instance_ids = prototype_instance_ids[valid_mask]
        similarity_matrix = torch.matmul(final_prototypes, final_prototypes.T) / self.temperature
        self.activation_instance['similarity_matrix']=prototype_instance_ids 
        instance_mask = torch.eq(final_instance_ids.unsqueeze(1), final_instance_ids.unsqueeze(0))
        label_mask = torch.eq(final_labels.unsqueeze(1), final_labels.unsqueeze(0))
        pos_mask = label_mask
        self.activation_instance['label_mask']=label_mask 
        pos_mask.fill_diagonal_(False)
        if pos_mask.sum() == 0: 
            return torch.tensor(0.0, device=features_all.device)

        log_prob = similarity_matrix - torch.log(torch.exp(similarity_matrix).sum(1, keepdim=True))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def instance_contrast_old(self, feat_l1, feat_l2, feat_u1, feat_u2,
                         pred_l1, pred_l2, pred_u1, pred_u2):
        """Instance-level contrastive learning"""
        # Global pooling
        instance_feat_l1 = F.adaptive_avg_pool2d(feat_l1, 1).squeeze(-1).squeeze(-1)
        instance_feat_l2 = F.adaptive_avg_pool2d(feat_l2, 1).squeeze(-1).squeeze(-1)
        instance_feat_u1 = F.adaptive_avg_pool2d(feat_u1, 1).squeeze(-1).squeeze(-1)
        instance_feat_u2 = F.adaptive_avg_pool2d(feat_u2, 1).squeeze(-1).squeeze(-1)
        
        # Project and normalize
        proj_l1 = F.normalize(self.instance_projector(instance_feat_l1), dim=1)
        proj_l2 = F.normalize(self.instance_projector(instance_feat_l2), dim=1)
        proj_u1 = F.normalize(self.instance_projector(instance_feat_u1), dim=1)
        proj_u2 = F.normalize(self.instance_projector(instance_feat_u2), dim=1)
        
        B = proj_l1.shape[0]
        
        # Get instance labels (dominant class)
        labels_l1 = pred_l1.argmax(dim=1).flatten(1).mode(dim=1)[0]  # B
        labels_l2 = pred_l2.argmax(dim=1).flatten(1).mode(dim=1)[0]
        labels_u1 = pred_u1.argmax(dim=1).flatten(1).mode(dim=1)[0]
        labels_u2 = pred_u2.argmax(dim=1).flatten(1).mode(dim=1)[0]
        
        # Reshape for contrastive loss (B, C, 1)
        proj_l1 = proj_l1.unsqueeze(2)
        proj_l2 = proj_l2.unsqueeze(2)
        proj_u1 = proj_u1.unsqueeze(2)
        proj_u2 = proj_u2.unsqueeze(2)
        
        # Compute contrastive losses
        loss_l1 = self.contrastive_loss(proj_l1, proj_l2, proj_u1, proj_u2,
                                       labels_l1, labels_l2, labels_u1, labels_u2)
        loss_l2 = self.contrastive_loss(proj_l2, proj_l1, proj_u1, proj_u2,
                                       labels_l2, labels_l1, labels_u1, labels_u2)
        loss_u1 = self.contrastive_loss(proj_u1, proj_u2, proj_l1, proj_l2,
                                       labels_u1, labels_u2, labels_l1, labels_l2)
        loss_u2 = self.contrastive_loss(proj_u2, proj_u1, proj_l1, proj_l2,
                                       labels_u2, labels_u1, labels_l1, labels_l2)
        
        return (loss_l1 + loss_l2 + loss_u1 + loss_u2)


# Short alias (the module is called PRI in the paper).
PRI = PixelRegionInstanceContrast
