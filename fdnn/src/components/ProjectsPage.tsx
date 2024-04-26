import { Routes, Route } from "react-router-dom";
import { ProjectGrid } from "./ProjectGrid";
import * as React from "react";

export interface IAugmentations {
    zoom_height_factor: number;
    zoom_width_factor: number;
    flip_horizontal: number;
    flip_vertical: number;
    random_rotation_factor: number;
}

export interface IProjectData {
    name: string;
    max_images: number;
    total_test: number;
    train_counts: { [key: string]: number };
    test_counts: { [key: string]: number };
    training_path: string;
    testing_path: string;
    seed: number;
    image_width: number;
    image_height: number;
    image_channels: number;
    labels: Array<string>;
    augmentations: IAugmentations;
}

export interface IProjectsPageState {
    projects: Array<IProjectData>;
}

export function ProjectsPage() {
    const [data, setdata] = React.useState<IProjectsPageState>({
        projects: [],
    });

    React.useEffect(() => {
        fetch("/api/project/").then((res) =>
            res.json().then((data: IProjectsPageState) => {
                setdata(data);
            })
        );
    }, []);

    const refresh = (callback?: () => void) => {
        fetch("/api/project/").then((res) =>
            res
                .json()
                .then((data: IProjectsPageState) => {
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
                <Route path="/" element={<ProjectGrid projects={data.projects} refresh={refresh} />} />
            </Routes>
        </div>
    );
}
