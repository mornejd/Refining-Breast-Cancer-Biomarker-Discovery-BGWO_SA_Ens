Sgenes <- read.table("C:\\Users\\....\\Sgenes45827.xls", sep="\t", header=TRUE, row.names=1)
PHgenes <- read.table("C:\\Users\\....\\PHgenes45827.xls", sep="\t", header=TRUE, row.names=1)

library(pheatmap)
library(ggplot2)
library(ggrepel)
library(limma)

design <- model.matrix(~0+sample.levels,PHgenes)
colnames(design) <- c("Cancer")
design <- cbind(Normal = 1 - design[, "Cancer"], design)

rownames(design) <- PHgenes$sample.labels

fit <- lmFit(Sgenes, design)
#head(fit$coefficients)
contrasts <- makeContrasts(Cancer - Normal, levels=design)
fit2 <- contrasts.fit(fit, contrasts)
fit2 <- eBayes(fit2, 0.01)

table_GSE <- topTable(fit2, adjust="fdr", sort.by="B", number = Inf,confint = TRUE)


hist(table_GSE$P.Value, col = brewer.pal(3, name = "Set2")[1],
     main = "MBC vs PBC - GSEmerge", xlab = "p-values")
hist(table_GSE$logFC, col = brewer.pal(3, name = "Set2")[1],
     main = "MBC vs PBC - GSEmerge", xlab = "logFC")
hist(table_GSE$adj.P.Val, col = brewer.pal(3, name = "Set2")[1],
     main = "MBC vs PBC - GSEmerge", xlab = "Adj.p.val")

GSEmerge_DEG_Results <- data.frame(Symbols = rownames(table_GSE),
                                   Log2FC= table_GSE$logFC,
                                   pvalue=table_GSE$P.Value,
                                   adj.pvalue=table_GSE$adj.P.Val,
                                   CI.L=table_GSE$CI.L,
                                   CI.R = table_GSE$CI.R, 
                                   t=table_GSE$t,stringsAsFactors = FALSE)


EnhancedVolcano(GSEmerge_DEG_Results,
                lab = as.character(GSEmerge_DEG_Results$Symbols),
                x = 'Log2FC',
                y = 'adj.pvalue',
                title="Volcano Plot Normal vs Breast Cancer",            
                xlab = bquote(~Log[2]~ "fold change"),
                ylab = bquote(~-Log[10]~adjusted~italic(P)),
                pCutoff = 0.05,
                FCcutoff = 2,
                legendPosition = "bottom",
                legendLabels=c("NS","Log2 FC","Adjusted p-value","Adjusted p-value & Log2 FC"))

sigGenes <- GSEmerge_DEG_Results[ GSEmerge_DEG_Results$adj.pvalue < 0.05 & 
                                    !is.na(GSEmerge_DEG_Results$pvalue) &
                                    abs(GSEmerge_DEG_Results$Log2FC) > 2, ]


UpReg <- sigGenes[sigGenes$adj.pvalue < 0.05 & 
                    sigGenes$Log2FC > 2, ]

DownReg <- sigGenes[sigGenes$adj.pvalue < 0.05 & 
                      sigGenes$Log2FC < 2, ]


SSS<- sigGenes$Symbols
FinalSet<-Sgenes[rownames(Sgenes) %in% SSS, ]            # Applying %in%-operator
FinalSet_t <- t(FinalSet)
Final <- merge(FinalSet_t, PHgenes, by = "row.names")       # Applying merge() function

write.table(Final, file=paste0("C:\\Users\\....\\Final.xls"), sep="\t", quote=FALSE)
write.table(UpReg, file=paste0("C:\\Users\\....\\UpReg.xls"), sep="\t", quote=FALSE)
write.table(DownReg, file=paste0("C:\\Users\\....\\DownReg.xls"), sep="\t", quote=FALSE)


MMM<- GSEmerge_DEG_Results$Symbols
MLSet<-Sgenes[rownames(Sgenes) %in% MMM, ]            # Applying %in%-operator
MLSet_t <- t(MLSet)
ML <- merge(MLSet_t, PHgenes, by = "row.names")       # Applying merge() function

write.table(ML, file=paste0("C:\\Users\\....\\ML.xls"), sep="\t", quote=FALSE)
