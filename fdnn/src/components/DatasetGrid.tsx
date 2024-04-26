import DatasetCard from "./DatasetCard";
import { DatasetCreate } from "./DatasetCreate";
import { IDataset } from "./DatasetPage";

interface IDatasetGridProps {
    datasets: Array<IDataset>;
    refresh: () => void;
}

export function DatasetGrid(props: IDatasetGridProps) {
    return (
        <div
            style={{
                margin: 15,
                textAlign: "left",
            }}
        >
            <DatasetCreate refreshDatasets={props.refresh} />
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
                {props.datasets.map((dataset) => (
                    <DatasetCard dataset={dataset} key={dataset.name} />
                ))}
            </div>
        </div>
    );
}
