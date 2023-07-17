# author: Nicolai Meyerhöfer

library(DAseq)
library(rjson)
library(reticulate)
library(stringr)
pd <- import("pandas")

#' Constructs a data frame comprised of all given samples.
#'
#' @param csv_paths CSV paths of all used samples
#' @param labels Labels of samples to use in given csv files
#' @returns data frame comprised of all given samples
read_data <- function(pickle_paths) {
  df <- data.frame()  # Create an empty data frame

  for (pickle_path in pickle_paths) {
    str_list = str_split(pickle_paths, "/") 
    label = basename(pickle_path)
    pickle_data <- pd$read_pickle(pickle_path)  # Read pickle file using reticulate
    pickle_data$label <- rep(label, nrow(pickle_data))
    df <- rbind(df, pickle_data)
    }
  return(df)
}

#' Calculates the individual differentially abundant (DA) cells and clusters for all of the given samples vs the given controls
#'
#' @param controls Dataframe of controls
#' @param patients Dataframe of patients
#' @param save_path Path to save results to
#' @param save_file_name Optional extra filename for saving DA cells and regions
#' @param remove_nrows Optional removal of rows of patient dataframe for validation
#' 
calculate_DA_clusters <- function(controls, patients, save_path = "./", json_save_path = "./", save_file_name = ""){

    runFItSNE <- function(data, seed.use = 0, reduction.name = "tsne", reduction.key = "tSNE_",
                            fast.R.path = "/data_slow/je30bery/DAseq-master/FIt-SNE/fast_tsne.R", ...)
    {
        current.dir <- getwd()
        source(fast.R.path, chdir = T)
        X.out <- fftRtsne(X = data, rand_seed = seed.use, ...)
        setwd(current.dir)
        return(X.out)
    }


    ids_controls <- controls[,ncol(controls)]
    ids_patients <- patients[,ncol(patients)]
    

    for(sample_id in ids_patients[!duplicated(ids_patients)]){
        print(sample_id)
        df_patient <- patients[patients$label == sample_id,]
        df_new_tmp <- rbind(controls, df_patient)
        df_new <- df_new_tmp[,1:ncol(df_new_tmp)-1]
        df_new <- sapply(df_new, as.numeric)
        
        print(dim(df_new))

        df.scaled <- scale(df_new[,1:ncol(df_new_tmp)-1])
        data_tsne <- runFItSNE(df.scaled)
        data_tsne.col <- ncol(data_tsne)
 
        colnames(data_tsne) <- paste("tSNE_", c(1:data_tsne.col), sep="")
        rownames(data_tsne) <- rownames(df.scaled)

        cell_labels <- as.character(df_new_tmp[, ncol(df_new_tmp)])
        print(cell_labels)
        labels_1 <- as.character(ids_patients[!duplicated(ids_patients)])
        labels_2 <- as.character(ids_controls[!duplicated(ids_controls)])
        
        print(labels_1)
        print(labels_2)


        df_scaled <- as.data.frame(df.scaled)
        

        da_cells <- getDAcells(
                X = df_scaled,
                cell.labels = cell_labels,
                labels.1 = labels_1,
                labels.2 = labels_2,
                k.vector = seq(50, 500, 50),
                plot.embedding = data_tsne
        )

        da_cells <- updateDAcells(
                X = da_cells,
                pred.thres = c(-0.8,0.8),
                do.plot = T,
                plot.embedding = data_tsne,
                size = 0.1
        )

        f <- paste(save_path, "da_cells_", save_file_name, sample_id, ".rds", sep="")
        saveRDS(da_cells, f)

        da_regions <- getDAregion(
                X = df.scaled,
                da.cells = da_cells,
                cell.labels = df_new_tmp[,ncol(df_new_tmp)],
                labels.1 = ids_patients[!duplicated(ids_patients)],
                labels.2 = ids_controls[!duplicated(ids_controls)],
                resolution = 0.01,
                min.cell = 50,
                plot.embedding = data_tsne,
                size = 0.1
        )

        f <- paste(save_path, "da_regions_", save_file_name, sample_id, ".rds", sep="")
        saveRDS(da_regions, f)

        # writing the json file
        da_cluster_name <- sample_id
        cluster_labels <- which(da_regions$DA.stat[,1] < 0)
        
        da_cluster_purities <- list(da_regions$DA.stat[cluster_labels])	
        da_cluster_cells_tmp <- list()
        da_cluster_sizes_tmp <- c()
            for(cluster_label in cluster_labels){
            cells <- which(da_regions$da.region.label == cluster_label)
            # indices -1 to fit python indices because R counts indices from 1 :D 
            for(i in 1:length(cells)){
                cells[i] <- cells[i] - 1
            }
            da_cluster_cells_tmp <- append(da_cluster_cells_tmp, list(cells))
            da_cluster_sizes_tmp <- c(da_cluster_sizes_tmp, length(cells))
        }
            da_cluster_sizes <- list(da_cluster_sizes_tmp)	
        da_cluster_cells <- list(da_cluster_cells_tmp)
        json_df <- data.frame(da_cluster_name)
        json_clusters <- data.frame(Sizes = I(da_cluster_sizes), Purities = I(da_cluster_purities), DA_Cells = I(da_cluster_cells))
        json_df$Clusters <- json_clusters
        json_file <- toJSON(json_df)
        write(json_file, file = paste(json_save_path, sample_id, "_clusters.json", sep="", collapse=NULL))
    }
}

# Get command line arguments
#args <- commandArgs(trailingOnly=TRUE)

# Check that at least 4 arguments were supplied
# stopifnot(length(args) >= 4)

# Parse command line arguments
control_dir <- "/data_slow/je30bery/spatial_proteomics/marker_expression_nuclei_results/control"
patient_dir <- "/data_slow/je30bery/spatial_proteomics/marker_expression_nuclei_results/case"


#control_samples <- strsplit(args[3], ",")[[1]] #TODO
#patient_samples <- strsplit(args[4], ",")[[1]] #TODO

r_file_out <- "/data_slow/je30bery/spatial_proteomics/DAseq_results/"
json_file_out <- "/data_slow/je30bery/spatial_proteomics/DAseq_results/"

# Check that the directories exist
if (!file.exists(control_dir) || !file.info(control_dir)$isdir) {
    stop(paste0("Directory does not exist: ", control_dir))
}
if (!file.exists(patient_dir) || !file.info(patient_dir)$isdir) {
    stop(paste0("Directory does not exist: ", patient_dir))
}

# Get CSV filenames
control_files <- list.files(control_dir, "*.pkl", full.names=TRUE)
patient_files <- list.files(patient_dir, "*.pkl", full.names=TRUE)

# Read data
control_data <- read_data(control_files)
if (is.null(control_data)) {
    warning(paste0("No pickle files found in ", control_dir))
}
patient_data <- read_data(patient_files)
if (is.null(patient_data)) {
    warning(paste0("No pickle files found in ", patient_dir))
}

# Calculate clusters
calculate_DA_clusters(controls = control_data, patients = patient_data,
                       save_path = r_file_out, json_save_path = json_file_out)