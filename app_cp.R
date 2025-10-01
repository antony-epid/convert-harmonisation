################################################################################
# Section 1. Load packages and data set
################################################################################

# Load the required packages
library(igraph)
library(dplyr)
library(metafor)
library(shiny)
library(DT)
library(visNetwork)
library(synthpop)
library(mvmeta)
library(metamisc)
library(lme4)
library(ggplot2)
library(nlme)
library(splines)
library(tidyr)
library(segmented)
library(mgcv)
library(xgboost)

# Load data outside of the server function
all_partic_data <- read.csv("uploaded_files/models_by_particpant_and_device_pair_BINS.csv")

# Check and remove relationships with negative correlation
all_partic_data <- subset(all_partic_data, r > 0.0)

# Create filtered data set
all_building <- data.frame(
  from = all_partic_data$predictor,
  to = all_partic_data$outcome,
  unit_from = all_partic_data$p_units,
  unit_to = all_partic_data$o_units,
  r = all_partic_data$r,
  person_minutes = all_partic_data$n,
  id = all_partic_data$id,
  from_mean = all_partic_data$predictor_mean,
  from_sd = all_partic_data$predictor_sd,
  to_mean = all_partic_data$outcome_mean,
  to_sd = all_partic_data$outcome_sd,
  predictor_p99_5 = all_partic_data$predictor_p99_5,
  outcome_p99_5 = all_partic_data$outcome_p99_5,
  sex = all_partic_data$sex,
  age = all_partic_data$age,
  study = all_partic_data$study,
  bmi = all_partic_data$bmi,
  region = all_partic_data$region,
  X_mean_0 = all_partic_data$bin_mean_0,
  Y_mean_0 = all_partic_data$bin_fit_0,
  Y_se_0 = all_partic_data$bin_fit_se_0,
  X_mean_1 = all_partic_data$bin_mean_1,
  Y_mean_1 = all_partic_data$bin_fit_1,
  Y_se_1 = all_partic_data$bin_fit_se_1,
  X_mean_2 = all_partic_data$bin_mean_2,
  Y_mean_2 = all_partic_data$bin_fit_2,
  Y_se_2 = all_partic_data$bin_fit_se_2,
  X_mean_3 = all_partic_data$bin_mean_3,
  Y_mean_3 = all_partic_data$bin_fit_3,
  Y_se_3 = all_partic_data$bin_fit_se_3,
  X_mean_4 = all_partic_data$bin_mean_4,
  Y_mean_4 = all_partic_data$bin_fit_4,
  Y_se_4 = all_partic_data$bin_fit_se_4,
  X_mean_5 = all_partic_data$bin_mean_5,
  Y_mean_5 = all_partic_data$bin_fit_5,
  Y_se_5 = all_partic_data$bin_fit_se_5,
  X_mean_6 = all_partic_data$bin_mean_6,
  Y_mean_6 = all_partic_data$bin_fit_6,
  Y_se_6 = all_partic_data$bin_fit_se_6,
  X_mean_7 = all_partic_data$bin_mean_7,
  Y_mean_7 = all_partic_data$bin_fit_7,
  Y_se_7 = all_partic_data$bin_fit_se_7,
  X_mean_8 = all_partic_data$bin_mean_8,
  Y_mean_8 = all_partic_data$bin_fit_8,
  Y_se_8 = all_partic_data$bin_fit_se_8,
  X_mean_9 = all_partic_data$bin_mean_9,
  Y_mean_9 = all_partic_data$bin_fit_9,
  Y_se_9 = all_partic_data$bin_fit_se_9,
  X_mean_10 = all_partic_data$bin_mean_10,
  Y_mean_10 = all_partic_data$bin_fit_10,
  Y_se_10 = all_partic_data$bin_fit_se_10
)

# set seed for random variables
set.seed(1986)

# Drop any pairs with <5 participants
all_building <- all_building %>%
  group_by(from, to) %>%
  mutate(pair_count = n())

all_building <- all_building %>%
  filter(pair_count >= 5)

# Unique devices and units for drop down lists
devices <- unique(x = all_building$from)
device_and_units <- all_building[!duplicated(all_building$from), ]
device_and_units <- subset(device_and_units, select = c(from, unit_from))
Predictor <- data.frame(Predictor = device_and_units$from, Predictor_units = device_and_units$unit_from)
Outcome <- data.frame(Outcome = device_and_units$from, Outcome_units = device_and_units$unit_from)



################################################################################
# Section 2. Set up the UI for the Shiny app
################################################################################

# Build the graph of the entire network
network <- graph_from_data_frame(all_building, directed = TRUE, vertices = devices)

# Load the current data
current_data <- all_partic_data

# Load the template for user contributions
template <- read.csv("uploader/upload_template.csv", header = TRUE)

# Define UI ----
ui <- fluidPage(
  titlePanel("Harmonisation for time-series physical activity data (beta)"),

  # Create a tabsetPanel with three tabs
  tabsetPanel(
    tabPanel(
      "Data overview",
      titlePanel(""),
      mainPanel(
        h5("Network graph of device pairs."),
        visNetworkOutput("overview_plot", width = "100%", height = "60vh"),
        DTOutput("summary_table", width = "100%")
      )
    ),
    tabPanel(
      "Mapping generator",
      titlePanel(""),
      sidebarLayout(
        sidebarPanel(
          actionButton("goButton", "Click to generate"),
          br(),
          HTML("<br><br>"),
          selectInput("start_variable", "Harmonising from:",
            selected = "Axivity NDW ENMO",
            choices = unique(all_building$from), multiple = FALSE
          ),
          selectInput("target_variable", "Harmonising to:",
            selected = "Actiheart branched equation PAEE",
            choices = unique(all_building$from), multiple = FALSE
          ),
          checkboxInput("include_men", "Men", value = TRUE),
          checkboxInput("include_women", "Women", value = TRUE),
          sliderInput("age", "Age (years)", min = 0, max = 100, value = c(0, 100)),
          sliderInput("bmi", "Body Mass Index", min = 0, max = 70, value = c(0, 70)),
          selectizeInput("region", "Select region(s):",
            choices = c(
              "All" = "", "Australia and Oceania", "Caribbean", "Central America", "Central & South Asia",
              "Eastern Europe", "Middle East", "Northern Africa", "Northern Europe",
              "North America", "Northeastern Asia", "South America", "Southern Africa",
              "Southern Europe", "Southeastern Asia", "Western Europe"
            ),
            multiple = TRUE
          ),
          selectInput("study", "Select studies:",
            choices = c("All" = "", unique(all_building$study)),
            multiple = TRUE
          ),
          numericInput("edges_max", "Maximum path length", 1, min = 1, max = NA, step = 1, width = NULL),
          numericInput("minutes_min", "Minimum minutes per person", 1440, min = 1, max = NA, step = 1, width = NULL),
          numericInput("decimals", "Significant figures", 4, min = 4, max = NA, step = 1, width = NULL),
          checkboxInput("exclude_direct", "Omit direct path", value = FALSE),
          selectInput("exclude_nodes", "Omit nodes:", choices = unique(all_building$from), selected = NULL, multiple = TRUE)
        ),
        mainPanel(
          visNetworkOutput("metan_network_plot", width = "100%", height = "60vh"),
          h3("Network mapping"),
          tableOutput("overall_mapping"),
          h3("Path-level mapping"),
          DTOutput("meta_table_routes", width = "100%"),
          h3("Edge-level mapping"),
          DTOutput("meta_table", width = "100%"),
          verbatimTextOutput("error_message")
        )
      )
    ),
    tabPanel(
      "Upload your mapping equations",
      sidebarLayout(
        sidebarPanel(
          fileInput("file1", "Upload your CSV File", accept = ".csv"),
          checkboxInput("header", "First row as column headers", TRUE),
          downloadButton("downloadData", "Download the CSV template")
        ),
        mainPanel(
          tableOutput("contents")
        )
      )
    ),
  )
)

# Run the app ----
server <- function(input, output, session) {
  
  
  
  ################################################################################
  # Section 2. Set up the server function for tab 1 of the Shiny app
  ################################################################################

    # Create table1
  table1 <- all_partic_data %>%
    dplyr::select(
      outcome, o_device, o_location, o_construct, o_units,
      predictor, p_device, p_location, p_construct, p_units,
      study, sex, age, bmi, n, r
    )

  table1$participants <- 1


  result <- table1 %>%
    group_by(
      outcome, o_device, o_location, o_construct, o_units,
      predictor, p_device, p_location, p_construct, p_units,
      study
    ) %>%
    summarize(
      percent_women = as.integer(round(mean(sex) * 100)),
      mean_age = mean(age),
      sd_age = sd(age),
      mean_bmi = mean(bmi),
      sd_bmi = sd(bmi),
      person_minutes = format(sum(n), big.mark = ","),
      participants = as.integer(round(sum(participants))),
      avg_fisher_z = sum(0.5 * log((1 + r) / (1 - r)) * n) / sum(n)
    )

  result$average_r <- round((exp(2 * result$avg_fisher_z) - 1) / (exp(2 * result$avg_fisher_z) + 1), digits = 2)

  result$agemeansd <- sprintf("%.0f (%.0f)", result$mean_age, result$sd_age)
  result$bmimeansd <- sprintf("%.0f (%.0f)", result$mean_bmi, result$sd_bmi)

  result$Sorted_Columns <- apply(result, 1, function(row) paste(pmin(row["predictor"], row["outcome"]), pmax(row["predictor"], row["outcome"]), collapse = ","))

  result <- result %>%
    group_by(Sorted_Columns) %>%
    dplyr::slice(1) %>%
    ungroup()

  result$Sorted_Columns <- NULL
  result <- result[order(result$study, result$predictor, result$outcome), ]

  predictor_details <- table1 %>%
    rename(
      id = predictor,
      device = p_device,
      wear_location = p_location,
      construct = p_construct,
      units = p_units
    ) %>%
    ungroup() %>%
    dplyr::select(id, device, wear_location, construct, units)

  outcome_details <- table1 %>%
    rename(
      id = outcome,
      device = o_device,
      wear_location = o_location,
      construct = o_construct,
      units = o_units
    ) %>%
    ungroup() %>%
    dplyr::select(id, device, wear_location, construct, units)

  device_details <- bind_rows(predictor_details, outcome_details) %>%
    distinct(id, .keep_all = TRUE) %>%
    dplyr::select(id, device, wear_location, construct, units)

  output$overview_plot <- renderVisNetwork({
    nodes <- data.frame(
      id = device_details$id,
      label = device_details$id,
      group = device_details$wear_location,
      font.size = 80,
      title = paste(
        "Device(s):", device_details$device,
        "<br>Wear Location(s):", device_details$wear_location,
        "<br>Construct:", device_details$construct,
        "<br>Units:", device_details$units,
        sep = " "
      )
    )

    edges <- data.frame(
      from = result$predictor,
      to = result$outcome,
      title = paste(
        "r=", result$average_r,
        "<br>n=", result$participants,
        "<br>Person-minutes=", result$person_minutes,
        sep = ""
      )
    )

    visNetwork(nodes, edges, height = "1000px", width = "100%") %>%
      visInteraction(dragNodes = TRUE, dragView = TRUE, zoomView = TRUE) %>%
      visPhysics(
        solver = "forceAtlas2Based",
        forceAtlas2Based = list(gravitationalConstant = -500),
        stabilization = list(enabled = TRUE, onlyDynamicEdges = FALSE)
      ) %>%
      visLayout(improvedLayout = TRUE) %>%
      visOptions(
        selectedBy = list(variable = "group", multiple = TRUE),
        highlightNearest = list(enabled = TRUE, hover = TRUE),
        nodesIdSelection = list(enabled = TRUE)
      )
  })

  output$summary_table <- DT::renderDataTable({
    fortable1 <- result %>%
      mutate(
        pearson = format(round(average_r, 2), nsmall = 2)
      ) %>%
      ungroup() %>%
      dplyr::select(
        "Study" = study,
        "Device 1" = predictor,
        "Device 2" = outcome,
        "Participants (n)" = participants,
        "Women (%)" = percent_women,
        "Mean (SD) age" = agemeansd,
        "Mean (SD) BMI" = bmimeansd,
        "Person-minutes" = person_minutes,
        "Pearson's r" = pearson
      )

    datatable(
      fortable1,
      options = list(
        ordering = TRUE,
        searching = TRUE,
        autoWidth = FALSE,
        lengthMenu = c(10, 25, 50, -1),
        pageLength = -1,
        columnDefs = list(
          list(targets = "_all", className = "dt-center"),
          list(targets = 0, width = "10px"),
          list(targets = 1, width = "180px"),
          list(targets = 2, width = "650px"),
          list(targets = 3, width = "650px"),
          list(targets = 4, width = "30px"),
          list(targets = 5, width = "40px"),
          list(targets = 6, width = "130px"),
          list(targets = 7, width = "130px"),
          list(targets = 8, width = "80px"),
          list(targets = 9, width = "40px")
        ),
        paging = FALSE,
        style = "font-size: 10px; white-space: nowrap;"
      )
    )
  })

  

  ################################################################################
  # Section 2. Set up the server function for tab 2 of the Shiny app
  ################################################################################

  ### Section 2A. Create the "edges" data frame containing data for all edges
  ### in the network between device1 and device2 defined by the user. This is
  ### achieved by filtering the data set based on user inputs, and using igraph
  ### to construct a network graph and then pull out the edges

  # Respond to "Go" button press (not quite working properly at the moment, only works on first use)
  observeEvent(input$goButton, {
    # Initialise device1 and 2
    device1 <- NULL
    device2 <- NULL

    # Create reactive variables for filtering
    filtered_network_data <- reactive({
      # Filter criteria based on user input
      min_age <- input$age[1]
      max_age <- input$age[2]
      min_bmi <- input$bmi[1]
      max_bmi <- input$bmi[2]
      include_men <- input$include_men
      include_women <- input$include_women
      selected_regions <- input$region
      selected_studies <- input$study
      min_minutes <- input$minutes_min
      excluded_nodes <- input$exclude_nodes

      # Filtering logic based on user input
      if (include_men == TRUE & include_women == FALSE) {
        criteria <- E(network)[
          (age >= min_age & age <= max_age & !is.na(age) & bmi >= min_bmi & bmi <= max_bmi & sex == 0) &
            (is.null(selected_regions) | length(selected_regions) == 0 | region %in% selected_regions) &
            (is.null(selected_studies) | length(selected_studies) == 0 | study %in% selected_studies) &
            person_minutes >= min_minutes & !is.na(person_minutes)
        ]
      } else if (include_men == FALSE & include_women == TRUE) {
        criteria <- E(network)[
          (age >= min_age & age <= max_age & !is.na(age) & bmi >= min_bmi & bmi <= max_bmi & sex == 1) &
            (is.null(selected_regions) | length(selected_regions) == 0 | region %in% selected_regions) &
            (is.null(selected_studies) | length(selected_studies) == 0 | study %in% selected_studies) &
            person_minutes >= min_minutes & !is.na(person_minutes)
        ]
      } else if (include_men == TRUE & include_women == TRUE) {
        criteria <- E(network)[
          (age >= min_age & age <= max_age & !is.na(age) & bmi >= min_bmi & bmi <= max_bmi) &
            (is.null(selected_regions) | length(selected_regions) == 0 | region %in% selected_regions) &
            (is.null(selected_studies) | length(selected_studies) == 0 | study %in% selected_studies) &
            person_minutes >= min_minutes & !is.na(person_minutes)
        ]
      }

      validate(
        need(
          input$include_men != FALSE | input$include_women != FALSE,
          "Please select at least one sex"
        )
      )

      filtered_network <- subgraph.edges(graph = network, eids = criteria)

      # Convert the filtered network to a data frame/tibble for counting combinations
      filtered_network_df <- as_tibble(get.data.frame(filtered_network, what = "edges"))

      # Count unique combinations and filter
      filtered_network_df <-
        filtered_network_df %>%
        group_by(from, to) %>%
        mutate(participants_with_data = n()) %>%
        ungroup() %>%
        filter(participants_with_data >= 20)

      # Recreate the network with the filtered edges
      filtered_network <- graph_from_data_frame(filtered_network_df, directed = TRUE)


      if (!is.null(excluded_nodes) && length(excluded_nodes) > 0) {
        vertices_to_delete <- which(V(filtered_network)$name %in% excluded_nodes)
        filtered_network <- delete.vertices(filtered_network, vertices_to_delete)
      }
      return(filtered_network)
    })

    # Reactive element for the routes
    routes_simple <- reactive({
      Exclude_direct <- input$exclude_direct
      device1 <<- input$start_variable
      device2 <<- input$target_variable
      routes_simple <- all_simple_paths(filtered_network_data(), from = device1, to = device2, mode = "out", cutoff = input$edges_max)

      validate(
        need(length(routes_simple) > 0, "Please ask for a longer path")
      )
      if (Exclude_direct == TRUE) {
        routes_simple <- routes_simple[sapply(routes_simple, function(x) length(x) != 2)]
      }
      return(routes_simple)
    })


    # reactive element to generate the table of meta-analysed routes
    joined_data <- reactive({
      Exclude_direct <- input$exclude_direct
      eids <- lapply(routes_simple(), function(x) {
        edges <- cbind(head(x, -1)$name, x[-1]$name)
        output <- data.frame("From" = edges[, 1], "To" = edges[, 2])
        return(output)
      })

      eids2 <- distinct(as.data.frame(do.call(rbind, eids)))

      final_eids <- apply(eids2, MARGIN = 1, FUN = function(x) {
        E(filtered_network_data(), directed = TRUE)[x[1] %->% x[2]]
      })

      network_routes <- subgraph.edges(graph = filtered_network_data(), eids = unique(unlist(final_eids)))
      edges <- igraph::as_data_frame(network_routes, "edges")



      ### Section 2B. Set up the edges data frame for modelling relationships by
      ### switching from wide to long format

      # Create a new data frame with one row per edge containing summary statistics
      from_mean_sd__rmse_mean <- edges %>%
        group_by(from, to) %>%
        summarize(
          from_weighted_mean = sum(from_mean * person_minutes) / sum(person_minutes),
          to_weighted_mean = sum(to_mean * person_minutes) / sum(person_minutes),
          total_n = sum(person_minutes),
          from_variance_weighted_mean = sum((person_minutes - 1) * from_sd^2 + person_minutes * (from_mean - from_weighted_mean)^2) / (sum(person_minutes) - 1),
          to_variance_weighted_mean = sum((person_minutes - 1) * to_sd^2 + person_minutes * (to_mean - to_weighted_mean)^2) / (sum(person_minutes) - 1),
          from_combined_sd = sqrt(from_variance_weighted_mean),
          to_combined_sd = sqrt(to_variance_weighted_mean),
          participants = n(),
          max_X = mean(predictor_p99_5) + sd(predictor_p99_5) * 3,
          max_Y = mean(outcome_p99_5) + sd(outcome_p99_5) * 3
        )

      # Switch the wide format edges data frame to long format, meaning each
      # participant has 11 rows for each device pair. Each row contains
      # binned predictor (X) and outcome (Y) data from 0-10. Bin 0 contains all
      # non-positive values
      edges_long <- edges %>%
        pivot_longer(
          cols = starts_with("X_mean") | starts_with("Y_mean") | starts_with("Y_se"),
          names_to = c(".value", "bin"),
          names_pattern = "(.*)_(\\d+)"
        ) %>%
        mutate(bin = as.numeric(bin)) %>%
        arrange(id, bin)

      # Add the square and cube terms
      edges_long$X_mean_2 <- edges_long$X_mean^2
      edges_long$X_mean_3 <- edges_long$X_mean^3

      # Use regression to impute SE where it is 0
      m1 <- lm(edges_long$Y_se ~ 1 + edges_long$Y_mean)
      replacement_se <- coef(m1)["(Intercept)"]
      edges_long$Y_se[edges_long$Y_se == 0] <- replacement_se

      # Calculate inverse variance weighting
      edges_long$weights <- 1 / edges_long$Y_se^2



      ### Section 2C. Use XGBoost decision tree model to regress Y on X for each
      ### edge and save the model with a unique identifier in the model_list

      # Initialise the model_list
      model_list <- list()

      # Wrapper function to train and store models
      xgboost_wrapper <- function(X_mean, Y_mean, from, to, id, weights, model_list) {
        # Create a data frame for features
        X <- data.frame(X_mean = X_mean, id = id, weights = weights)
        y <- Y_mean

        x_matrix <- as.matrix(X[, !(names(X) %in% c("id", "weights"))])

        # Convert the data to DMatrix format (required by xgboost)
        dtrain <- xgb.DMatrix(
          data = x_matrix,
          label = y
        )

        # Set up the parameters for XGBoost model
        params_mod <- list(
          objective = "reg:squarederror", # Regression task
          eval_metric = "rmse", # Use RMSE as evaluation metric
          eta = 0.1, # Learning rate
          max_depth = 8, # Maximum depth of trees
          nrounds = 200 # Number of boosting rounds
        )

        # Train the XGBoost model
        model <- xgboost(params = params_mod, data = dtrain, nrounds = params_mod$nrounds, early_stopping_rounds = 10)

        # Store the model in the model_list using 'from' and 'to' as the key
        print(paste0("Storing model for from = ", from, ", to = ", to))

        key <- paste0(from, "_", to)
        model_list[[key]] <- model

        # Make predictions on the training data
        predictions <- predict(model, x_matrix)

        # Calculate residuals
        residuals <- y - predictions

        # Calculate RMSE
        rmse <- sqrt(mean(residuals^2))

        # Calculate R2
        rss <- sum((y - predictions)^2)
        tss <- sum((y - mean(y))^2)
        R2 <- 1 - (rss / tss)

        # Return both results and the updated model_list
        df <- data.frame(rmse = rmse, R2 = R2)
        return(list(df = df, model_list = model_list)) # Return the updated model_list
      }

      # Apply the wrapper function to each (from, to) group in edges_long
      meta_out <- edges_long %>%
        group_by(from, to) %>%
        group_modify(~ {
          # Call xgboost_wrapper and return both the results and the updated model_list
          result <- xgboost_wrapper(
            X_mean = .x$X_mean,
            Y_mean = .x$Y_mean,
            from = .y$from,
            to = .y$to,
            weights = .x$weights,
            id = .x$id,
            model_list = model_list
          )

          # Update model_list with the updated one returned from xgboost_wrapper
          model_list <<- result$model_list
          assign("model_list", model_list, envir = .GlobalEnv)
          return(result$df)
        })

      # Combine the model RMSE and R2 with edge-level summary stats, one row per edge
      meta_out <- left_join(meta_out, from_mean_sd__rmse_mean, by = c("from", "to"))

      # Generate some additional edge-level summary statistics

      # Average correlation
      avg_fisher_z_by_pair <- edges %>%
        group_by(from, to) %>%
        summarise(
          avg_fisher_z = sum(0.5 * log((1 + r) / (1 - r)) * person_minutes) / sum(person_minutes)
        ) %>%
        ungroup()

      avg_correlation_by_pair <- avg_fisher_z_by_pair %>%
        mutate(avg_correlation = (exp(2 * avg_fisher_z) - 1) / (exp(2 * avg_fisher_z) + 1)) %>%
        dplyr::select(from, to, avg_correlation)

      # Proportion men and women, mean age, number of participants, person-minutes
      other_sums <- edges %>%
        left_join(avg_correlation_by_pair, by = c("from" = "from", "to" = "to")) %>%
        group_by(from, unit_from, to, unit_to) %>%
        summarise(
          sex = 100 * (round(mean(sex), 1)),
          age = mean(age),
          n = n(),
          "Person-minutes" = format(sum(person_minutes), big.mark = ","),
          r = first(avg_correlation)
        )

      joined_data <- left_join(meta_out, other_sums, by = c("from" = "from", "to" = "to"))

      if (Exclude_direct == TRUE) {
        joined_data <- joined_data[!(joined_data$from == device1 & joined_data$to == device2), ]
      }

      return(joined_data)
    })



    ### Section 2D. Set up the params data frame containing the routes and edges

    # Set the number of iterations for the calculation of output for each route
    # EG four routes results in 400 iterations in total
    sample_iterations <- 100

    # Create a list of data frames called "eids". Each data frame corresponds to a unique route
    # from device1 to device2. Each of these data frames has a "from" and "to" column
    # and each row represents a edge in that route. Row number corresponds to order
    # of edges in that route. Each data frame has same number of rows as the current
    # route has edges
    mcmc <- reactive({
      eids <- lapply(routes_simple(), function(x) {
        edges <- cbind(head(x, -1)$name, x[-1]$name)
        data.frame(from = edges[, 1], to = edges[, 2])
      })

    # Select the relevant variables from the edge summary data from previous
    # section and attach to each unique edge in the list of data frames "eids"
    # This creates "params" which has RMSE, mean and sd for the two devices
    # for each edge in each route, number of participants, and the theoretical
    # max predictor and outcome values for each edge
      params <- lapply(eids, function(x) {
        joined <- dplyr::left_join(x, joined_data()[, c("from", "to", "rmse", "from_weighted_mean", "from_combined_sd", "to_weighted_mean", "to_combined_sd", "participants", "max_X", "max_Y")],
          by = c("from", "to")
        )
        return(joined)
      })

    # Add a column for a unique identifier for ecah edge
      combineColumns <- function(df) {
        df$combined <- paste(df$from, df$to, sep = "_")
        return(df)
      }

      params <- lapply(params, combineColumns)

    # Add the current route length
      add_route_length <- function(df) {
        df$path_length <- length(df$from)
        return(df)
      }

      params <- lapply(params, add_route_length)

      # Add a numerical label for the route (the position of the route
      # data frame in the params list)
      params <- lapply(seq_along(params), function(i) {
        df <- params[[i]]
        df$source_df <- i
        return(df)
      })

      # Count the number of occurrences of each edge in the network. Edges in
      # routes of length 1 always receive a count of 1. Edges in routes of length 2
      # receive a count for each use in routes of length 2, plus the use in the
      # direct route length 1 if it is used. This continues for longer route lengths.

      countOccurrences <- function(params) {
        df <- do.call(rbind, params)
        df$edgename <- paste(df$from, df$to)
        df <- df %>% arrange(path_length)
        df <- df %>%
          group_by(edgename) %>%
          mutate(
            count = sapply(path_length, function(x) sum(path_length <= x & edgename == edgename))
          ) %>%
          ungroup()

        df_list_recombined <- split(df, df$source_df)
        return(df_list_recombined)
      }

      params <- countOccurrences(params)

      # Expand the RMSE so that edges receive a penalty for re-use
      expandRMSE <- function(df) {
        df$rmse_mean_exp <- df$rmse * sqrt(df$count)
        return(df)
      }

      params <- lapply(params, expandRMSE)

      

      ### Section 2E. Generate the route-level equations

      # Set the number of observations in the simulated distribution
      reps <- 1000

      # Function for generating the origin distribution.
      func_origin_dist <- function(x) {
        # parameters for the normal distribution
        mean <- as.numeric(x["from_weighted_mean"])
        sd <- as.numeric(x["from_combined_sd"])

        # parameters for the lognormal distribution
        mu <- log(mean^2 / sqrt(sd^2 + mean^2))
        sigma <- sqrt(log(1 + (sd^2 / mean^2)))

        # generate log normal data
        log_normal_data <- rlnorm(reps, meanlog = mu, sdlog = sigma)
        log_normal_data
      }

      # Functions to pull out the winsorisation values
      func_max_X <- function(x) as.numeric(x["max_X"])
      func_max_Y <- function(x) as.numeric(x["max_Y"])

      # Functions for generating noise based on RMSE
      func_sigma <- function(mean_value, x) {
        return(log(1 + (as.numeric(x["rmse_mean_exp"]) / mean_value)))
      }

      # Function to adjust the noise
      func_adjusted_noise <- function(adjusted_noise, x) {
        ratio <- as.numeric(x["rmse_mean_exp"]) / sd(adjusted_noise)
        if (ratio >= 1) {
          adjusted_noise * (sd(adjusted_noise) / (as.numeric(x["rmse_mean_exp"])))
        } else if (ratio < 1) {
          adjusted_noise * ((as.numeric(x["rmse_mean_exp"]) / sd(adjusted_noise)))
        }
      }

      # Define function to take the current input data, make a prediction, add
      # noise, and save the output. x refers to the current row (edge) in the current
      # data frame in the params list of route-level data frames (y)
      calculate_output <- function(estimate, x, model_list) {
        # Open the current input data and ensure it is numeric for prediction
        estimate_vec <- as.numeric(estimate[, 1])

        # Retrieve the model for the given (from, to) pair using 'combined' as the key
        # from the model list
        key <- paste0(x$combined)
        model <- model_list[[key]]

        # Predict output using the current model from XGBoost
        output <- predict(model, as.matrix(estimate_vec))
        output_mean <- mean(output)

        # Calculate sigma and noise based on positive mean
        dists_sigma <- func_sigma(output_mean, x)
        dists_mu <- -0.5 * dists_sigma^2
        normal_noise <- rnorm(reps, mean = dists_mu, sd = dists_sigma)
        lognormal_noise <- exp(normal_noise)

        output_noisy <- output * lognormal_noise
        noise_added <- output_noisy - output
        adjusted_noise_sd_correct <- func_adjusted_noise(noise_added, x)
        output_new <- output + adjusted_noise_sd_correct

        return(output_new)
      }



      # Define function to generate the origin distribution
      step_fun <- function(y, route_number, seed, model_list) {
        set.seed(seed)

        # Generate the origin distribution using the first row ([1, ]) in the current
        # dataframe in params (y)
        dist_origin <- as.data.frame(func_origin_dist(y[1, ]))
        colnames(dist_origin) <- "dist_origin"

        # Save the winsorisation value for the current route
        MAX_X <- func_max_X(y[1, ])
        #  dist_origin$dist_origin[dist_origin$dist_origin > MAX_X] <- MAX_X

        # Initialize the estimate variable for the first leg
        estimate <- as.data.frame(dist_origin)

        # Initialize data frame to hold output from each leg
        dist_output <- as.data.frame(matrix(0, nrow = reps, ncol = 1))

        # Set the current route length using number of rows in current data frame
        legs <- nrow(y)

        # For each data frame (y) in params, travel down the rows (edges) applying calculate
        # output each time until we reach nrows
        for (i in 1:legs) {
          # Calculate the output for each leg, passing model_list to calculate_output

          # Set the current row (edge) number for clarity
          current_x <- y[i, ]

          # Call calculate output defined above
          dist_output <- calculate_output(estimate, current_x, model_list)

          # Reset the estimate for the next leg
          estimate <- as.data.frame(dist_output)
        }

        # Combine the final output with the origin data, plus the route number
        # and winsorisation value
        output_df <- data.frame(dist_origin, estimate, route_number, MAX_X)
        colnames(output_df) <- c("dist_origin", "estimate", "route_number", "MAX_X")

        return(output_df)
      }

      # Call step_fun for each data frame in params. Each data frame represents
      # a route. Save the output_df for each route and each iteration.
      # This creates a list of dataframes. Each data frame in the list represents
      #  an iteration. Length of list is number of iterations. Each data frame contains
      # input data, output data, route number and winsorisation value for each route.
      # For example, with four routes and 1000 observations, each dataframe will have
      # 4000 rows.
      output_list <- vector("list", sample_iterations)
      for (i in 1:sample_iterations) {
        seed <- i
        output_list[[i]] <- do.call(rbind, lapply(seq_along(params), function(j) step_fun(params[[j]], j, seed, model_list = model_list)))
      }
      all_output <- output_list
      return(all_output)
    })

    
    
    ### Section 2F. Run the route-level models

    # Create a data frame to hold the string route description and the coefficients
    routes_df <- reactive({
      dp <- input$decimals
      routes_df <- data.frame(Route = character(0), Segment1BETA = numeric(0), Segment1SE = numeric(0), Segment2BETA = numeric(0), Segment2SE = numeric(0), Segment3BETA = numeric(0), Segment3SE = numeric(0), RMSE = numeric(0))

      for (route in routes_simple()) {
        path <- paste(V(filtered_network_data())$name[route], collapse = " -> ")
        routes_df <- rbind(routes_df, data.frame(Route = path))
      }

      mcmc_data_list <- mcmc()

      # Fit the model for each route/
      fit_model <- function(route_number, mcmc_data) {
        # Select the data for current route
        subset_df <- mcmc_data[mcmc_data$route_number == route_number, ]
        subset_df <- subset_df[!is.nan(subset_df$estimate), ]

        # Create the weight to deal with unequal variances
        subset_df$weight_X <- 1 / (subset_df$dist_origin)
        subset_df$weight_X_norm <- subset_df$weight_X / max(subset_df$weight_X)

        # Fit the polynomial model with weights
        poly_route <- lm(estimate ~ 0 + dist_origin + I(dist_origin^2) + I(dist_origin^3), data = subset_df, weights = weight_X)

        # Save the coefficiens and RMSE
        vcov_mat <- vcov(poly_route)
        predictions <- predict(poly_route, subset_df)
        residuals <- subset_df$estimate - predictions
        observed_values <- subset_df$estimate

        rss <- sum((observed_values - predictions)^2)
        tss <- sum((observed_values - mean(observed_values))^2)
        R2 <- 1 - (rss / tss)
        rmse_route <- sqrt(mean(residuals^2))

        betas <- coef(poly_route)
        ses <- sqrt(diag(vcov(poly_route)))
        end_betaA <- betas[1]
        end_betaA_se <- ses[1]
        end_betaB <- betas[2]
        end_betaB_se <- ses[2]
        end_betaC <- betas[3]
        end_betaC_se <- ses[3]

        return(data.frame(
          BETA1 = end_betaA,
          SE1 = end_betaA_se,
          BETA2 = end_betaB,
          SE2 = end_betaB_se,
          BETA3 = end_betaC,
          SE3 = end_betaC_se,
          RMSE = rmse_route
        ))
      }

      # Call the function for each route in each iteration
      all_step_dist_betaA_betaB_list <- lapply(mcmc_data_list, function(mcmc_data) {
        do.call(rbind, lapply(unique(mcmc_data$route_number), fit_model, mcmc_data))
      })

      # create new dataframe to store the average across iterations
      result_df <- all_step_dist_betaA_betaB_list[[1]]
      result_df[, ] <- 0

      # Iterate over each dataframe and each cell to compute the mean
      for (df in all_step_dist_betaA_betaB_list) {
        result_df <- result_df + df
      }

      # Divide by the number of dataframes to get the mean
      result_df <- result_df / sample_iterations
      step_dist_betaA_betaB <- result_df

      # Save the overall result for each route
      routes_df$BETA1 <- step_dist_betaA_betaB[, 1] # store the beta
      # routes_df$Segment1SE <- step_dist_betaA_betaB[, 2] # store the SE for betaA
      routes_df$BETA2 <- step_dist_betaA_betaB[, 3] # store the intercept
      # routes_df$Segment2SE <- step_dist_betaA_betaB[, 4]  # store the SE for betaB
      routes_df$BETA3 <- step_dist_betaA_betaB[, 5] # store the intercept
      # routes_df$Segment3SE <- step_dist_betaA_betaB[, 6]  # store the SE for betaC
      routes_df$RMSE <- step_dist_betaA_betaB[, 7]
      # routes_df$R_squared <- step_dist_betaA_betaB[,11]

      routes_df
    })

    mcmc_updated <- reactive({
      RMSE_routes <- data.frame(RMSE = routes_df()$RMSE, route_number = seq_len(nrow(routes_df())))
      updated_mcmc <- lapply(mcmc(), function(df) {
        df <- merge(df, RMSE_routes, by = "route_number", all.x = TRUE)
        df$weight <- 1 / (df$dist_origin)
        return(df)
      })
      return(updated_mcmc)
    })

    # create the table of route-level results for Shiny
    output$meta_table_routes <- DT::renderDataTable({
      dp <- input$decimals
      display_data <- routes_df() %>%
        mutate(
          BETA1 = formatC(signif(BETA1, dp), format = "g"),
          # Segment1SE = format(round(Segment1SE, dp), nsmall = dp),
          BETA2 = formatC(signif(BETA2, dp), format = "g"),
          # Segment2SE = format(round(Segment2SE, dp), nsmall = dp),
          BETA3 = formatC(signif(BETA3, dp), format = "g"),
          # Segment3SE = format(round(Segment3SE, dp), nsmall = dp),
          RMSE = formatC(signif(RMSE, dp), format = "g"),
          # "R squared" = format(round(R_squared, dp), nsmall = dp)
        ) %>%
        dplyr::select(Route, BETA1, BETA2, BETA3, RMSE)

      datatable(
        display_data,
        options = list(
          ordering = TRUE, # Enable sorting
          searching = TRUE, # Enable search/filter
          autoWidth = FALSE,
          lengthMenu = c(10, 25, 50, -1), # -1 to show all rows
          pageLength = -1, # -1 to show all rows
          paging = FALSE,
          style = "font-size: 10px; white-space: nowrap;", # Prevent text wrapping
          columnDefs = list(list(targets = "_all", className = "dt-center"))
        )
      )
    })
    
    
    
    ### Section 2G. Run the overall network model

    # Create the table to display meta-result
    output$overall_mapping <- renderTable({
      dp <- input$decimals

      process_dataframe <- function(df) {
        # Set the number of routes
        max_routes <- length(unique(df$route_number))

        # Create the weight to deal with unequal variances
        df$weight_X <- 1 / (df$dist_origin * df$RMSE^2)
        df$weight_X_norm <- df$weight_X / max(df$weight_X)

        # Set the winsorisation value across all routes
        Max_Predictor <- max(df$MAX_X)

        # Run the model if number of routes is one.
        if (max_routes == 1) {
          poly_meta <- lm(estimate ~ 0 + dist_origin + I(dist_origin^2) + I(dist_origin^3), data = df, weights = weight_X_norm)


          # Save the coefficiens and RMSE
          vcov_mat <- vcov(poly_meta)
          predictions <- predict(poly_meta, df)
          residuals <- df$estimate - predictions
          observed_values <- df$estimate

          rss <- sum((observed_values - predictions)^2)
          tss <- sum((observed_values - mean(observed_values))^2)
          R2 <- 1 - (rss / tss)
          rmse_route <- sqrt(mean(residuals^2))

          betas <- coef(poly_meta)
          ses <- sqrt(diag(vcov(poly_meta)))

          # Run multi-level model if there are multiple routes
        } else if (max_routes > 1) {
          # Ensure polynomial terms are created beforehand
          df$dist_origin_sq <- df$dist_origin^2
          df$dist_origin_cu <- df$dist_origin^3

          # Fit mixed model
          poly_meta <- lmer(estimate ~ 0 + dist_origin + dist_origin_sq + dist_origin_cu + (1 | route_number),
            data = df,
            weights = weight_X_norm
          )

          # Save the coefficiens and RMSE
          vcov_mat <- vcov(poly_meta)
          predictions <- predict(poly_meta, df)
          residuals <- df$estimate - predictions
          observed_values <- df$estimate

          rss <- sum(residuals^2)
          tss <- sum((observed_values - mean(observed_values))^2)
          R2 <- 1 - (rss / tss)
          rmse_route <- sqrt(mean(residuals^2))

          betas <- fixef(poly_meta)
          ses <- sqrt(diag(vcov_mat))
        } else {
          stop("Please increase maximum route length")
        }

        # Save and format the betas and ses
        BETA1 <- as.numeric(format(round(betas[1], 10), nsmall = 10))
        BETA2 <- as.numeric(format(round(betas[2], 10), nsmall = 10))
        BETA3 <- as.numeric(format(round(betas[3], 10), nsmall = 10))

        SE1 <- as.numeric(format(round(ses[1], 10), nsmall = 10))
        SE2 <- as.numeric(format(round(ses[2], 10), nsmall = 10))
        SE3 <- as.numeric(format(round(ses[3], 10), nsmall = 10))

        list(
          # Intercept =  Intercept,
          BETA1 = BETA1,
          # Segment1SE = Segment1SE,
          BETA2 = BETA2,
          # Segment2SE = Segment2SE,
          BETA3 = BETA3,
          # Segment3SE = Segment3SE,
          RMSE = rmse_route,
          Max_Predictor = Max_Predictor
          # R_squared = R2
        )
      }


      # Run the model on each dataframe and save results for each iteration
      results <- lapply(mcmc_updated(), process_dataframe)


      # Calculate and average across iterations
      calculate_average <- function(lists, element_name) {
        element_values <- sapply(lists, function(lst) lst[[element_name]])
        mean(element_values, na.rm = TRUE)
      }

      # Store the average network results
      results_av <- sapply(names(results[[1]]), function(name) calculate_average(results, name))

      # Create the table for the network result
      summary_data <- data.frame(
        Predictor = c(device1),
        Outcome = c(device2),
        # Intercept = format(round(results_av["Intercept"], dp), nsmall = dp),
        BETA1 = formatC(signif(results_av[["BETA1"]], dp), format = "g"),
        # Segment1SE = format(round(results_av["Segment1SE"], dp), nsmall = dp),
        BETA2 = formatC(signif(results_av[["BETA2"]], dp), format = "g"),
        # Segment2SE = format(round(results_av["Segment2SE"], dp), nsmall = dp),
        BETA3 = formatC(signif(results_av[["BETA3"]], dp), format = "g"),
        RMSE = formatC(signif(results_av[["RMSE"]], dp), format = "g"),
        Max_Predictor = format(round(results_av["Max_Predictor"], dp), nsmall = dp)
        # Segment2SE = format(round(results_av["Segment2SE"], dp), nsmall = dp),
        # "R squared" = format(round(results_av["R_squared"], dp), nsmall = dp)
      )
    })

    

    ### Section 2H. Edge level tale and rendering the network figure

    # Create the table of edge-level betas/ses
    output$meta_table <- DT::renderDataTable({
      dp <- input$decimals
      renamed_data <- joined_data() %>%
        rename(
          Age = age,
          n = n
        ) %>%
        mutate(
          Age = round(Age, 1),
          "Women (%)" = format(round(sex, 1), nsmall = 1),
          Predictor = paste(from, " (", unit_from, ")", sep = ""),
          Outcome = paste(to, " (", unit_to, ")", sep = "")
        ) %>%
        ungroup() %>%
        dplyr::select(Predictor, Outcome, n, "Person-minutes", "Women (%)", Age)

      # hide data if <5
      replace_na_columns <- c("Age", "Women (%)")
      renamed_data <- renamed_data %>%
        mutate(across(all_of(replace_na_columns), ~ ifelse(n < 5, "NA", .)))

      datatable(
        renamed_data,
        options = list(
          ordering = TRUE, # Enable sorting
          searching = TRUE, # Enable search/filter
          autoWidth = FALSE,
          lengthMenu = c(10, 25, 50, -1), # -1 to show all rows
          pageLength = -1, # -1 to show all rows
          paging = FALSE,
          style = "font-size: 10px; white-space: nowrap;", # Prevent text wrapping
          columnDefs = list(list(targets = "_all", className = "dt-center"))
        )
      )
    })

    # Render the metanetwork plot
    output$metan_network_plot <- renderVisNetwork({
      device1_value <- device1
      device2_value <- device2

      joined_data_2 <- joined_data()

      # Plot the routes
      device_list <- joined_data_2 %>%
        dplyr::select(from, to)

      all_details <- as.vector(as.matrix(device_list))

      device_list <- unique(all_details)

      nodes <- data.frame(id = device_list, label = device_list) %>%
        left_join(device_details, by = "id")

      nodes1 <- nodes %>%
        mutate(start_finish = case_when(
          id == device1_value ~ 1,
          id == device2_value ~ 2,
          TRUE ~ 0
        )) %>%
        mutate(id = as.character(id))

      nodes <- data.frame(
        id = nodes1$id, label = nodes1$id, group = nodes1$start_finish,
        size = 20, font.size = 20,
        title = paste("Device(s):", nodes1$device,
          "<br>Wear Location(s):", nodes1$wear_location,
          "<br>Construct:", nodes1$construct,
          "<br>Units:", nodes1$units,
          sep = " "
        )
      )

      edges <- data.frame(
        from = joined_data_2$from, to = joined_data_2$to,
        title = paste("r=", format(round(joined_data_2$r, 2), nsmall = 2),
          "<br>n=", joined_data_2$n,
          "<br>Person-minutes=", joined_data_2$`Person-minutes`,
          sep = ""
        ),
        arrows = c("middle;to")
      )

      # Create a shape vector based on 'nodes$start_finish' values
      shapes <- ifelse(nodes1$start_finish == 2, "star",
        ifelse(nodes1$start_finish == 1, "diamond", "dot")
      )

      # Assign colors and shapes to nodes
      nodes$shape <- shapes
      visNetwork(nodes, edges, height = "1000px", width = "100%") %>%
        visInteraction(dragNodes = TRUE, dragView = TRUE, zoomView = TRUE) %>%
        visPhysics(
          solver = "forceAtlas2Based",
          forceAtlas2Based = list(gravitationalConstant = -400)
        ) %>%
        visLayout(improvedLayout = TRUE) %>%
        visNodes() %>%
        visOptions(
          highlightNearest = list(enabled = TRUE, hover = TRUE),
          nodesIdSelection = list(enabled = TRUE)
        )
    })

    
    
    ################################################################################
    # Section 3. Set up the server function for tab 3 of the Shiny app
    ################################################################################

    # This section is incomplete. It will be used for uploading new data

    output$contents <- renderTable({
      file <- input$file1
      ext <- tools::file_ext(file$datapath)

      req(file)
      validate(expr = {
        if (ext != "csv") {
          error_message <- "Data not uploaded. Please upload a CSV file"
          stop(error_message)
        }
      })
      # Read the uploaded file
      uploaded_data <- read.csv(file$datapath, header = input$header)

      # Append the uploaded data to the current data
      combined_data <- rbind(current_data, uploaded_data)

      # Save the combined data to the current_data.csv file
      write.csv(combined_data, "uploaded_files/models_by_particpant_and_device_pair.csv", row.names = FALSE)



      # Return the uploaded data
      uploaded_data
    })

    output$downloadData <- downloadHandler(
      filename = function() {
        paste0("upload_template.csv")
      },
      content = function(file) {
        write.csv(template, file, row.names = FALSE)
      }
    )
  })
} # CLOSE THE SERVER LOGIC

shinyApp(ui, server)
