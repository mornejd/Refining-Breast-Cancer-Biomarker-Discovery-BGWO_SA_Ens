require(GEOquery)
require(hgu133a.db)
require(hgu133acdf)
require(annotate)
library(BiocParallel)
library(EnhancedVolcano)

BiocParallel::SerialParam()
Sys.setenv("VROOM_CONNECTION_SIZE"=10*131072)
my.gse <- "GSE10810"
if(!file.exists("geo_downloads")) dir.create("geo_downloads")
if(!file.exists("results"))  dir.create("results", recursive=TRUE)
##get data from GEO
my.geo.gse <- getGEO(GEO=my.gse, filename=NULL, destdir="./geo_downloads", GSElimits=NULL, GSEMatrix=TRUE, AnnotGPL=FALSE, getGPL=FALSE)
##explore structure of data
##get rid of list structure
my.geo.gse <- my.geo.gse[[1]]
##object is now an ExpressionSet
class(my.geo.gse)
pData(my.geo.gse)$data_processing[1]

if(!file.exists(paste0("./geo_downloads/",my.gse)))
  getGEOSuppFiles(my.gse, makeDirectory=T, baseDir="geo_downloads")
##list files in working directory
list.files("geo_downloads")
##function creates a directory and downloads the "tarball" that contains
##gzipped CEL files. We need to untar the tarball but we do not need to unzip
##the CEL files. R can read them in compressed format.
untar(paste0("geo_downloads/",my.gse,"/",my.gse,"_RAW.tar"), exdir=paste0("geo_downloads/",my.gse,"/CEL"))
list.files(paste0("geo_downloads/",my.gse,"/CEL"))
my.cels <- list.files(paste0("geo_downloads/",my.gse,"/CEL"), pattern=".CEL")
my.cels <- sort(my.cels)
my.cels
##make data frame of phenoData
my.pdata <- as.data.frame(pData(my.geo.gse), stringsAsFactors=F)
head(my.pdata)
my.pdata <- my.pdata[, c("title", "geo_accession","tumor (t) vs healthy (s):ch1")]
colnames(my.pdata)[colnames(my.pdata) == "tumor (t) vs healthy (s):ch1"] <- "Label"
my.pdata <- my.pdata[order(rownames(my.pdata)), ]
head(my.pdata, 10)
#################
temp.rownames <- paste(my.cels, sep="")
table(temp.rownames == my.cels)
rownames(my.pdata) <- temp.rownames
rm(temp.rownames)
table(rownames(my.pdata) == my.cels)
my.pdata
#################
write.table(my.pdata, file=paste0("geo_downloads/",my.gse,"/CEL/",my.gse,"_SelectPhenoData.txt"), sep="\t", quote=F)
#################
##repform affy normalization
library(affy)
cel.path <- paste0("geo_downloads/",my.gse,"/CEL")
my.affy <- ReadAffy(celfile.path=cel.path, phenoData=paste(cel.path, paste0(my.gse,"_SelectPhenoData.txt"), sep="/"))
show(my.affy)
#################
table(pData(my.affy)$Label)
##create shorter descriptive levels and labels
# Create a character vector of labels with commas
##create shorter descriptive levels and labels
labels <- c(rep("0", 27),
            rep("1", 31))
# Assign the labels to the desired column in your data frame
pData(my.affy)$sample.levels <- labels
pData(my.affy)$sample.labels <- labels

table(pData(my.affy)$sample.levels)
##quick visual comparison of pData
cbind(as.character(pData(my.affy)$Label), pData(my.affy)$sample.levels, pData(my.affy)$sample.labels)
##Calculate gene level expression measures
my.rmaN <- rma(my.affy, normalize=F, background=F)
head(exprs(my.rmaN))
##make sample.levels a factor making 0 the reference.
pData(my.rmaN)$sample.levels <- as.factor(pData(my.rmaN)$sample.levels)
pData(my.rmaN)$sample.levels <- relevel(pData(my.rmaN)$sample.levels, ref="0")
levels(pData(my.rmaN)$sample.levels)
##Load limma to get plotDensities function. Load RColorBrewer to provide good
##color palettes.
library(limma)
library(RColorBrewer)
##show palettes available with RColorBrewer
display.brewer.all()
level.pal <- brewer.pal(6, "Dark2")
level.pal
##sample.levels is a factor. unname will change this to integer data
##to easily assign colors to the various levels.
level.cols <- level.pal[unname(pData(my.rmaN)$sample.levels)]
level.cols
##Density plot of rma values show need for normalization
plotDensities(exprs(my.rmaN), legend=F, col=level.cols, main="Before RMA")
legend("topright", legend=levels(pData(my.rmaN)$sample.levels), fill=level.pal)
pdf(file="results/DensityNoNorm.pdf", w=6, h=6)
dev.off()
##Boxplot is another way of showing the same thing.
boxplot(exprs(my.rmaN), las=2, names=pData(my.rmaN)$sample.labels, outline=F, col=level.cols, main="a) Before RMA",cex.main=2, cex.lab=2, cex.axis=2)
pdf(file="results/DensityNoNorm2.pdf", w=6, h=6)
dev.off()

##Now use normalization and background correction
my.rma <- rma(my.affy)
plotDensities(exprs(my.rma), legend=F, col=level.cols, main="After RMA")
pdf(file="results/DensityNorm.pdf", w=6, h=6)
dev.off()

boxplot(exprs(my.rma), las=2, names=pData(my.rma)$sample.labels, outline=F, col=level.cols,cex.main=2, cex.lab=2, cex.axis=2)
pdf(file="results/DensityNorm2.pdf", w=6, h=6)
dev.off()

##save rma values
write.table(exprs(my.rma), file=paste0("results/",my.gse,"_RMA_Norm.xls"), sep="\t", quote=FALSE)
########################
my.calls <- mas5calls(my.affy)
head(exprs(my.calls))
##the next expression will work down the rows of the data matrix and 
##return the number of samples in which the gene has been called as present.
present <- apply(exprs(my.calls), 1, function(x)(sum(x == "P")))
head(present)
table(present)
prop.table(table(present)) * 100
plotDensities(exprs(my.rma)[present >= 1, ], col=level.cols, legend=F, main="Present >= 1")
plotDensities(exprs(my.rma)[present < 1, ], col=level.cols, legend=F, main="Present < 1")
##remove probesets for genes that are not expressed
genes <- exprs(my.rma)
pgenes0 <- genes[present >= 1, ]
pgenes1 <- pgenes0
pgenes2 <- pgenes1[-grep("AFFX", rownames(pgenes1)), ]

####Annotation
probes = row.names(pgenes2)
Symbols = unlist(mget(probes,hgu133aSYMBOL,ifnotfound=NA))
Entrez_IDs = unlist(mget(probes,hgu133aENTREZID,ifnotfound=NA))
pgenes3 = cbind(probes, Symbols,Entrez_IDs,pgenes2)
#Remove Genes with NA value
pgenes3 <- na.omit(pgenes3)


DF = as.data.frame(pgenes3)
dat <- as.data.frame(sapply(DF[,4:ncol(DF)], as.numeric))
Sgenes=limma::avereps(dat, ID = DF$Symbols)
Sgenes_DF = as.data.frame(Sgenes)
my.rma2 <- ExpressionSet(assayData=Sgenes,phenoData=my.rma@phenoData,protocolData=my.rma@protocolData,annotation=my.rma@annotation)
####Save final dataset
pData(my.rma2)$sample.levels <- as.factor(pData(my.rma2)$sample.levels)
pData(my.rma2)$sample.levels <- relevel(pData(my.rma2)$sample.levels, ref="0")

write.table(Sgenes, file=paste0("C:\\Users\\....\\Sgenes.xls"), sep="\t", quote=FALSE)

PHgenes <- pData(my.rma2)
write.table(PHgenes, file=paste0("C:\\Users\\....\\PHgenes.xls"), sep="\t", quote=FALSE)
