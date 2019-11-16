#!/usr/bin/ruby

# This script should be run on the result EXPORT from Microsoft VOTT app,
# which can be usually found in the vott-json-export folder
require 'json'

DATASET_NAME = "training-usbase"
VOTT_PROJECT_DIR = "./#{DATASET_NAME}/vott-json-export"
EXPORT_PATH_CLASSES = "./model_data/#{DATASET_NAME}.names"
EXPORT_PATH_REGIONS_TEST = "./model_data/#{DATASET_NAME}_test.txt"
EXPORT_PATH_REGIONS_TRAIN = "./model_data/#{DATASET_NAME}_train.txt"

# xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
# xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id
# Note: make sure that x_max < width and y_max < height

class ConvertVott2Yolo
    attr_reader :project_file

    def initialize
        # Find json
        @project_file = Dir.entries(VOTT_PROJECT_DIR).select { |e| e[".json"] }.first
        @metadata = JSON.parse(File.read(VOTT_PROJECT_DIR + "/" + @project_file))
    end

    def get_classes
        @classes = @metadata["tags"].map { |c| c["name"] }
        @class_to_id = @classes.map.with_index { |num,c| [num, c] }.to_h
        @id_to_class = @classes.map.with_index { |num,c| [c, num] }.to_h

        # Write classes to file
        File.open(EXPORT_PATH_CLASSES, "w") do |f|
            f.write(@classes.join("\n"))
        end
    end

    def convert
        get_classes

        assets_list = @metadata["assets"].keys

        converted_lines = []

        assets_list.each do |asset_id|
            regions = @metadata["assets"][asset_id]["regions"]
            next if regions == nil || regions.size == 0

            asset_tag_blocks = []

            regions.each do |region|
                region["tags"].each do |tag| # for each tag, add a line to converted_lines
                    class_id = @class_to_id[tag]
                    box = region["boundingBox"]
                    x_min = box["left"].round
                    y_min = box["top"].round
                    x_max = x_min + box["width"].round
                    y_max = y_min + box["height"].round
                    asset_tag_blocks << "#{x_min},#{y_min},#{x_max},#{y_max},#{class_id}"
                    # format: image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id
                    # example: xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
                end
            end

            filename = @metadata["assets"][asset_id]["asset"]["path"].gsub("file:", "")
            filename = filename.split("/")
            filename[filename.size - 2] += "/vott-json-export"
            filename = filename.join("/") + ".jpg"

            # NOTICE: post-process filename
            # for Z390 training
            filename = filename.gsub("/Users/denis/src/io.ultrasound.yolov3tf/", "/home/denis/src/io.ultrasound.yolov3tfkr/")
            converted_lines << "#{filename} #{asset_tag_blocks.join(' ')}"
        end

        @converted_lines = converted_lines

        # split converted lines into test and training
        # split converted lines into test and training
        distribute_training_test(converted_lines)
    end

    def distribute_training_test(lines)
        test_lines = @converted_lines.sample((@converted_lines.size * 0.2).to_i)
        File.open(EXPORT_PATH_REGIONS_TEST, "w") do |f|
            f.write(test_lines.join("\n"))
        end

        train_lines = @converted_lines.select { |l| !test_lines.include?(l) }
        File.open(EXPORT_PATH_REGIONS_TRAIN, "w") do |f|
            f.write(train_lines.join("\n"))
        end
    end
end

converter = ConvertVott2Yolo.new
converter.convert
