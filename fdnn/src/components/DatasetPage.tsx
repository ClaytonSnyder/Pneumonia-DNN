import * as React from "react";
import DatasetCard from "./DatasetCard";
import { DatasetCreate } from "./DatasetCreate";

export interface IKaggleData {
    identifier: string;
    url: string;
}

export interface IDataset {
    name: string;
    description: string;
    labels: Array<string>;
    total_image_count: number;
    kaggle_datasets_included: Array<IKaggleData>;
}

interface IContentState {
    datasets: Array<IDataset>;
}
export function DatasetPage() {
    const [data, setdata] = React.useState<IContentState>({
        datasets: [],
    });

    React.useEffect(() => {
        fetch("/api/dataset/").then((res) =>
            res.json().then((data: IContentState) => {
                setdata(data);
            })
        );
    }, []);

    const refresh = () => {
        fetch("/api/dataset/").then((res) =>
            res.json().then((data: IContentState) => {
                setdata(data);
            })
        );
    };

    return (
        <div
            style={{
                margin: 15,
                textAlign: "left",
            }}
        >
            <DatasetCreate refreshDatasets={refresh} />
            <div
                style={{
                    display: "flex",
                    flexWrap: "wrap",
                    overflow: "hidden",
                    width: "100%",
                    height: "100%",
                    position: "relative",
                    marginTop: 15,
                }}
            >
                {data.datasets.map((dataset) => (
                    <DatasetCard dataset={dataset} key={dataset.name} />
                ))}
            </div>
        </div>
    );
}
