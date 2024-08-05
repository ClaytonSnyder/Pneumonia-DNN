import * as React from "react";
import { Route, Routes } from "react-router-dom";
import { DatasetGrid } from "./DatasetGrid";
import { DetasetDetails } from "./DatasetDetails";

export interface IKaggleDataLabelMetadata {
    label: string;
    folder: string;
    alias: string;
    included: boolean;
}
export interface IKaggleData {
    identifier: string;
    url: string;
    path_to_metadata: string;
    label_column?: string;
    image_column: string;
    folder_column: string;
    folder_to_lower: boolean;
    labels: Array<IKaggleDataLabelMetadata>;
}

export interface IDataset {
    name: string;
    description: string;
    labels: { [key: string]: number };
    width_distribution: { [key: string]: number };
    height_distribution: { [key: string]: number };
    avg_height: number;
    avg_width: number;
    total_image_count: number;
    corrupt_image_count: number;
    kaggle_datasets_included: Array<IKaggleData>;
}

export interface IDatasetState {
    datasets: Array<IDataset>;
}
export function DatasetPage() {
    const [data, setdata] = React.useState<IDatasetState>({
        datasets: [],
    });

    React.useEffect(() => {
        fetch("/api/dataset/").then((res) =>
            res.json().then((data: IDatasetState) => {
                setdata(data);
            })
        );
    }, []);

    const refresh = (callback?: () => void) => {
        fetch("/api/dataset/").then((res) =>
            res
                .json()
                .then((data: IDatasetState) => {
                    setdata(data);
                })
                .then(() => {
                    if (callback) {
                        callback();
                    }
                })
        );
    };

    return (
        <div>
            <Routes>
                <Route path="/" element={<DatasetGrid refresh={refresh} datasets={data.datasets} />} />
                <Route path="/:id" element={<DetasetDetails refresh={refresh} datasets={data.datasets} />} />
            </Routes>
        </div>
    );
}
